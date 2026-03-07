//! Seqlock + 楽観ロック — 低レイテンシ並行性制御。
//!
//! 読み取り優先のシナリオ (時系列クエリ) で
//! ロックフリー読み取りを実現する Seqlock と、
//! 書き込み競合時にリトライする楽観ロックを提供。
//!
//! Author: Moroya Sakamoto

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, Ordering};

// ============================================================================
// Seqlock
// ============================================================================

/// Seqlock — ライター 1、リーダー N の高速同期プリミティブ。
///
/// シーケンス番号が奇数の間は書き込み中を示す。
/// リーダーは読み取り前後でシーケンス番号を比較し、
/// 変化していればリトライする。
pub struct SeqLock<T> {
    seq: AtomicU64,
    data: UnsafeCell<T>,
}

// SeqLock は書き込み側の排他を呼び出し側で保証する前提
unsafe impl<T: Send> Send for SeqLock<T> {}
unsafe impl<T: Send> Sync for SeqLock<T> {}

impl<T: Copy> SeqLock<T> {
    /// 新しい `SeqLock` を初期値で作成。
    pub const fn new(value: T) -> Self {
        Self {
            seq: AtomicU64::new(0),
            data: UnsafeCell::new(value),
        }
    }

    /// ロックフリー読み取り。
    ///
    /// 書き込み中だった場合は自動的にリトライする。
    /// 最大 `max_retries` 回試行し、すべて失敗した場合は
    /// 最後の読み取り値を返す。
    pub fn read(&self) -> T {
        loop {
            let seq1 = self.seq.load(Ordering::Acquire);
            if seq1 & 1 != 0 {
                // 書き込み中 → スピン
                core::hint::spin_loop();
                continue;
            }

            // SAFETY: seq が偶数の間は書き込みが行われない
            let value = unsafe { *self.data.get() };

            let seq2 = self.seq.load(Ordering::Acquire);
            if seq1 == seq2 {
                return value;
            }
            // シーケンスが変わった → リトライ
            core::hint::spin_loop();
        }
    }

    /// 書き込み。
    ///
    /// 排他ロックは呼び出し側で保証する必要がある
    /// (単一ライター or 外部ミューテックス)。
    pub fn write(&self, value: T) {
        // シーケンスを奇数にして「書き込み中」を通知
        self.seq.fetch_add(1, Ordering::Release);

        // SAFETY: 単一ライター保証
        unsafe {
            *self.data.get() = value;
        }

        // シーケンスを偶数に戻して「書き込み完了」を通知
        self.seq.fetch_add(1, Ordering::Release);
    }

    /// 現在のシーケンス番号を取得。
    #[must_use]
    pub fn sequence(&self) -> u64 {
        self.seq.load(Ordering::Relaxed)
    }
}

// ============================================================================
// 楽観ロック
// ============================================================================

/// 楽観ロックのバージョンスタンプ。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Version(pub u64);

/// 楽観ロック — Compare-And-Swap ベースの並行性制御。
///
/// 読み取り時にバージョンを取得し、書き込み時にバージョンが
/// 変わっていなければ成功。変わっていれば競合として失敗。
pub struct OptimisticLock {
    version: AtomicU64,
}

/// 楽観ロックのエラー。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimisticError {
    /// バージョン不一致 (他のライターが先に書き込んだ)。
    VersionConflict { expected: Version, actual: Version },
}

impl std::fmt::Display for OptimisticError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VersionConflict { expected, actual } => {
                write!(
                    f,
                    "Version conflict: expected {}, actual {}",
                    expected.0, actual.0
                )
            }
        }
    }
}

impl std::error::Error for OptimisticError {}

impl OptimisticLock {
    /// 新しい楽観ロックを作成。
    #[must_use]
    pub const fn new() -> Self {
        Self {
            version: AtomicU64::new(0),
        }
    }

    /// 現在のバージョンを取得。
    #[must_use]
    pub fn read_version(&self) -> Version {
        Version(self.version.load(Ordering::Acquire))
    }

    /// バージョンを比較してインクリメント (CAS)。
    ///
    /// 成功時は新しいバージョンを返す。
    /// 失敗時は `VersionConflict` を返す。
    ///
    /// # Errors
    ///
    /// 他のライターが先にバージョンを更新していた場合。
    pub fn compare_and_increment(&self, expected: Version) -> Result<Version, OptimisticError> {
        match self.version.compare_exchange(
            expected.0,
            expected.0 + 1,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => Ok(Version(expected.0 + 1)),
            Err(actual) => Err(OptimisticError::VersionConflict {
                expected,
                actual: Version(actual),
            }),
        }
    }

    /// 強制的にバージョンをインクリメント (無条件)。
    pub fn force_increment(&self) -> Version {
        let prev = self.version.fetch_add(1, Ordering::AcqRel);
        Version(prev + 1)
    }

    /// 現在のバージョン番号を取得。
    #[must_use]
    pub fn current_version(&self) -> u64 {
        self.version.load(Ordering::Relaxed)
    }
}

impl Default for OptimisticLock {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// 楽観トランザクション
// ============================================================================

/// 楽観トランザクション — 読み取りセットのバージョンを検証。
///
/// 複数キーを読み取り、書き込み時に全キーのバージョンが
/// 変わっていないことを検証する。
pub struct OptimisticTxn {
    /// 読み取りセット: (キー, 読み取り時バージョン)。
    read_set: Vec<(u64, Version)>,
    /// 書き込みセット: (キー, 値)。
    write_set: Vec<(u64, f32)>,
}

impl OptimisticTxn {
    /// 新しいトランザクションを開始。
    #[must_use]
    pub const fn new() -> Self {
        Self {
            read_set: Vec::new(),
            write_set: Vec::new(),
        }
    }

    /// 読み取りセットにエントリを追加。
    pub fn record_read(&mut self, key: u64, version: Version) {
        self.read_set.push((key, version));
    }

    /// 書き込みセットにエントリを追加。
    pub fn record_write(&mut self, key: u64, value: f32) {
        self.write_set.push((key, value));
    }

    /// 読み取りセットのサイズ。
    #[must_use]
    pub const fn read_set_size(&self) -> usize {
        self.read_set.len()
    }

    /// 書き込みセットのサイズ。
    #[must_use]
    pub const fn write_set_size(&self) -> usize {
        self.write_set.len()
    }

    /// 読み取りセットを検証。
    ///
    /// `version_lookup` は各キーの現在バージョンを返す関数。
    /// 全てのキーのバージョンが読み取り時と一致していれば `Ok`。
    ///
    /// # Errors
    ///
    /// いずれかのキーのバージョンが変わっていた場合。
    pub fn validate<F>(&self, version_lookup: F) -> Result<(), OptimisticError>
    where
        F: Fn(u64) -> Version,
    {
        for &(key, expected_version) in &self.read_set {
            let current = version_lookup(key);
            if current != expected_version {
                return Err(OptimisticError::VersionConflict {
                    expected: expected_version,
                    actual: current,
                });
            }
        }
        Ok(())
    }

    /// 書き込みセットを取得して消費。
    #[must_use]
    pub fn into_writes(self) -> Vec<(u64, f32)> {
        self.write_set
    }
}

impl Default for OptimisticTxn {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// テスト
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seqlock_read_write() {
        let lock = SeqLock::new(42u64);
        assert_eq!(lock.read(), 42);
        lock.write(100);
        assert_eq!(lock.read(), 100);
    }

    #[test]
    fn seqlock_sequence_increments() {
        let lock = SeqLock::new(0u32);
        assert_eq!(lock.sequence(), 0);
        lock.write(1);
        assert_eq!(lock.sequence(), 2); // +1 (start) +1 (end)
        lock.write(2);
        assert_eq!(lock.sequence(), 4);
    }

    #[test]
    fn seqlock_f32() {
        let lock = SeqLock::new(3.125_f32);
        assert!((lock.read() - 3.125).abs() < f32::EPSILON);
        lock.write(2.71);
        assert!((lock.read() - 2.71).abs() < f32::EPSILON);
    }

    #[test]
    fn seqlock_struct() {
        #[derive(Debug, Clone, Copy, PartialEq)]
        struct Point {
            x: f32,
            y: f32,
        }
        let lock = SeqLock::new(Point { x: 1.0, y: 2.0 });
        let p = lock.read();
        assert!((p.x - 1.0).abs() < f32::EPSILON);
        lock.write(Point { x: 3.0, y: 4.0 });
        let p2 = lock.read();
        assert!((p2.x - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn optimistic_lock_basic() {
        let lock = OptimisticLock::new();
        let v = lock.read_version();
        assert_eq!(v, Version(0));

        let new_v = lock.compare_and_increment(v).unwrap();
        assert_eq!(new_v, Version(1));
    }

    #[test]
    fn optimistic_lock_conflict() {
        let lock = OptimisticLock::new();
        let v1 = lock.read_version();
        lock.force_increment(); // 別のライター

        let result = lock.compare_and_increment(v1);
        assert!(result.is_err());
        if let Err(OptimisticError::VersionConflict { expected, actual }) = result {
            assert_eq!(expected, Version(0));
            assert_eq!(actual, Version(1));
        }
    }

    #[test]
    fn optimistic_lock_force_increment() {
        let lock = OptimisticLock::new();
        let v1 = lock.force_increment();
        assert_eq!(v1, Version(1));
        let v2 = lock.force_increment();
        assert_eq!(v2, Version(2));
    }

    #[test]
    fn optimistic_lock_default() {
        let lock = OptimisticLock::default();
        assert_eq!(lock.current_version(), 0);
    }

    #[test]
    fn optimistic_error_display() {
        let err = OptimisticError::VersionConflict {
            expected: Version(1),
            actual: Version(2),
        };
        let s = err.to_string();
        assert!(s.contains("Version conflict"));
        assert!(s.contains('1'));
        assert!(s.contains('2'));
    }

    #[test]
    fn optimistic_txn_basic() {
        let txn = OptimisticTxn::new();
        assert_eq!(txn.read_set_size(), 0);
        assert_eq!(txn.write_set_size(), 0);
    }

    #[test]
    fn optimistic_txn_record_and_validate() {
        let mut txn = OptimisticTxn::new();
        txn.record_read(1, Version(0));
        txn.record_read(2, Version(0));
        txn.record_write(1, 42.0);

        assert_eq!(txn.read_set_size(), 2);
        assert_eq!(txn.write_set_size(), 1);

        // バージョン一致 → 成功
        let result = txn.validate(|_key| Version(0));
        assert!(result.is_ok());
    }

    #[test]
    fn optimistic_txn_validate_conflict() {
        let mut txn = OptimisticTxn::new();
        txn.record_read(1, Version(0));
        txn.record_read(2, Version(0));

        // key=2 のバージョンが変わっている → 競合
        let result = txn.validate(|key| if key == 2 { Version(1) } else { Version(0) });
        assert!(result.is_err());
    }

    #[test]
    fn optimistic_txn_into_writes() {
        let mut txn = OptimisticTxn::new();
        txn.record_write(10, 1.0);
        txn.record_write(20, 2.0);
        let writes = txn.into_writes();
        assert_eq!(writes.len(), 2);
        assert_eq!(writes[0], (10, 1.0));
    }

    #[test]
    fn optimistic_txn_default() {
        let txn = OptimisticTxn::default();
        assert_eq!(txn.read_set_size(), 0);
    }

    #[test]
    fn seqlock_multiple_writes() {
        let lock = SeqLock::new(0i64);
        for i in 1..=100 {
            lock.write(i);
        }
        assert_eq!(lock.read(), 100);
        assert_eq!(lock.sequence(), 200); // 100 writes × 2
    }
}
