//! ACID トランザクション (MVCC + ロックマネージャ)
//!
//! スナップショット分離によるマルチバージョン並行制御と、
//! キー粒度の排他/共有ロック管理。

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};

/// トランザクション ID。
pub type TxnId = u64;

/// トランザクション状態。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxnStatus {
    /// 実行中。
    Active,
    /// コミット済み。
    Committed,
    /// アボート済み。
    Aborted,
}

/// MVCC バージョンエントリ。
#[derive(Debug, Clone)]
pub struct MvccEntry<T: Clone> {
    /// 書き込みトランザクション ID。
    pub txn_id: TxnId,
    /// 値。
    pub value: T,
    /// コミット時刻 (0 = 未コミット)。
    pub commit_ts: u64,
}

/// MVCC ストア (スナップショット分離)。
#[derive(Debug)]
pub struct MvccStore<T: Clone> {
    /// キー → バージョンチェーン (新しい順)。
    versions: HashMap<String, Vec<MvccEntry<T>>>,
    /// トランザクション状態。
    txn_status: HashMap<TxnId, TxnStatus>,
    /// トランザクション開始時刻。
    txn_start_ts: HashMap<TxnId, u64>,
    /// トランザクションの書き込みセット。
    txn_write_set: HashMap<TxnId, HashSet<String>>,
    /// 次のトランザクション ID。
    next_txn_id: AtomicU64,
    /// 単調増加タイムスタンプ。
    current_ts: AtomicU64,
}

impl<T: Clone> Default for MvccStore<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> MvccStore<T> {
    /// 新しい MVCC ストアを作成。
    #[must_use]
    pub fn new() -> Self {
        Self {
            versions: HashMap::new(),
            txn_status: HashMap::new(),
            txn_start_ts: HashMap::new(),
            txn_write_set: HashMap::new(),
            next_txn_id: AtomicU64::new(1),
            current_ts: AtomicU64::new(1),
        }
    }

    /// タイムスタンプを進める。
    fn advance_ts(&self) -> u64 {
        self.current_ts.fetch_add(1, Ordering::SeqCst)
    }

    /// トランザクションを開始。
    pub fn begin(&self) -> TxnId {
        self.next_txn_id.fetch_add(1, Ordering::SeqCst)
    }

    /// トランザクションを登録 (begin 後に呼ぶ)。
    pub fn register(&mut self, txn_id: TxnId) {
        let ts = self.advance_ts();
        self.txn_status.insert(txn_id, TxnStatus::Active);
        self.txn_start_ts.insert(txn_id, ts);
        self.txn_write_set.insert(txn_id, HashSet::new());
    }

    /// スナップショット読み取り。
    ///
    /// 開始時刻以前にコミットされた最新バージョンを返す。
    #[must_use]
    pub fn read(&self, txn_id: TxnId, key: &str) -> Option<T> {
        let start_ts = *self.txn_start_ts.get(&txn_id)?;
        let chain = self.versions.get(key)?;

        // 自トランザクションの書き込みを優先
        for entry in chain.iter().rev() {
            if entry.txn_id == txn_id {
                return Some(entry.value.clone());
            }
        }

        // スナップショット分離: 開始時刻以前のコミット済みバージョン
        for entry in chain.iter().rev() {
            if entry.commit_ts > 0
                && entry.commit_ts <= start_ts
                && self.txn_status.get(&entry.txn_id) == Some(&TxnStatus::Committed)
            {
                return Some(entry.value.clone());
            }
        }

        None
    }

    /// 書き込み。
    ///
    /// # Errors
    ///
    /// トランザクションが非アクティブ、または書き込み競合時。
    pub fn write(&mut self, txn_id: TxnId, key: String, value: T) -> Result<(), TxnError> {
        if self.txn_status.get(&txn_id) != Some(&TxnStatus::Active) {
            return Err(TxnError::NotActive);
        }

        // 書き込み-書き込み競合チェック
        if let Some(chain) = self.versions.get(&key) {
            for entry in chain.iter().rev() {
                if entry.txn_id != txn_id
                    && entry.commit_ts == 0
                    && self.txn_status.get(&entry.txn_id) == Some(&TxnStatus::Active)
                {
                    return Err(TxnError::WriteConflict);
                }
            }
        }

        let entry = MvccEntry {
            txn_id,
            value,
            commit_ts: 0,
        };

        self.versions.entry(key.clone()).or_default().push(entry);

        if let Some(ws) = self.txn_write_set.get_mut(&txn_id) {
            ws.insert(key);
        }

        Ok(())
    }

    /// コミット。
    ///
    /// # Errors
    ///
    /// トランザクションが非アクティブの場合。
    pub fn commit(&mut self, txn_id: TxnId) -> Result<u64, TxnError> {
        if self.txn_status.get(&txn_id) != Some(&TxnStatus::Active) {
            return Err(TxnError::NotActive);
        }

        let commit_ts = self.advance_ts();

        // 書き込みセットのバージョンにコミット時刻を付与
        if let Some(write_set) = self.txn_write_set.get(&txn_id).cloned() {
            for key in &write_set {
                if let Some(chain) = self.versions.get_mut(key) {
                    for entry in chain.iter_mut().rev() {
                        if entry.txn_id == txn_id && entry.commit_ts == 0 {
                            entry.commit_ts = commit_ts;
                        }
                    }
                }
            }
        }

        self.txn_status.insert(txn_id, TxnStatus::Committed);
        Ok(commit_ts)
    }

    /// アボート。
    ///
    /// # Errors
    ///
    /// トランザクションが非アクティブの場合。
    pub fn abort(&mut self, txn_id: TxnId) -> Result<(), TxnError> {
        if self.txn_status.get(&txn_id) != Some(&TxnStatus::Active) {
            return Err(TxnError::NotActive);
        }

        // 書き込みセットのバージョンを削除
        if let Some(write_set) = self.txn_write_set.get(&txn_id).cloned() {
            for key in &write_set {
                if let Some(chain) = self.versions.get_mut(key) {
                    chain.retain(|e| e.txn_id != txn_id);
                }
            }
        }

        self.txn_status.insert(txn_id, TxnStatus::Aborted);
        Ok(())
    }

    /// トランザクション状態を取得。
    #[must_use]
    pub fn status(&self, txn_id: TxnId) -> Option<TxnStatus> {
        self.txn_status.get(&txn_id).copied()
    }

    /// 内部バージョンチェーン数 (テスト用)。
    #[must_use]
    pub fn lock_count_internal(&self) -> usize {
        self.versions.len()
    }

    /// 古いバージョンをガベージコレクション。
    ///
    /// 指定時刻以前のアボート済み/不要バージョンを削除。
    pub fn gc(&mut self, before_ts: u64) -> usize {
        let mut removed = 0;

        for chain in self.versions.values_mut() {
            let before_len = chain.len();

            // コミット済みの最新バージョンより古い、かつ before_ts 以前のものを削除
            // ただしコミット済み最新は残す
            if chain.len() > 1 {
                let mut keep_from = 0;
                for (i, entry) in chain.iter().enumerate().rev() {
                    if entry.commit_ts > 0 && entry.commit_ts <= before_ts {
                        keep_from = i;
                        break;
                    }
                }
                if keep_from > 0 {
                    chain.drain(..keep_from);
                }
            }

            // アボート済みトランザクションのエントリを削除
            chain.retain(|e| self.txn_status.get(&e.txn_id) != Some(&TxnStatus::Aborted));

            removed += before_len - chain.len();
        }

        removed
    }
}

/// ロックモード。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockMode {
    /// 共有ロック (読み取り)。
    Shared,
    /// 排他ロック (書き込み)。
    Exclusive,
}

impl std::fmt::Display for LockMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shared => write!(f, "Shared"),
            Self::Exclusive => write!(f, "Exclusive"),
        }
    }
}

/// ロックエントリ。
#[derive(Debug)]
struct LockEntry {
    /// 保持中のトランザクション。
    holders: HashSet<TxnId>,
    /// ロックモード。
    mode: LockMode,
    /// 待機キュー。
    waiters: VecDeque<(TxnId, LockMode)>,
}

/// ロックマネージャ (キー粒度ロック)。
#[derive(Debug)]
pub struct LockManager {
    /// キー → ロック情報。
    locks: HashMap<String, LockEntry>,
    /// トランザクション → 保持キー。
    held_locks: HashMap<TxnId, HashSet<String>>,
}

impl Default for LockManager {
    fn default() -> Self {
        Self::new()
    }
}

impl LockManager {
    /// 新しいロックマネージャを作成。
    #[must_use]
    pub fn new() -> Self {
        Self {
            locks: HashMap::new(),
            held_locks: HashMap::new(),
        }
    }

    /// ロックを取得。
    ///
    /// # Errors
    ///
    /// ロック競合時。
    pub fn acquire(&mut self, txn_id: TxnId, key: &str, mode: LockMode) -> Result<(), TxnError> {
        let entry = self
            .locks
            .entry(key.to_string())
            .or_insert_with(|| LockEntry {
                holders: HashSet::new(),
                mode: LockMode::Shared,
                waiters: VecDeque::new(),
            });

        if entry.holders.is_empty() {
            // ロック未取得 → 即座に取得
            entry.holders.insert(txn_id);
            entry.mode = mode;
            self.held_locks
                .entry(txn_id)
                .or_default()
                .insert(key.to_string());
            return Ok(());
        }

        if entry.holders.contains(&txn_id) {
            // 同じトランザクションが既に保持
            if mode == LockMode::Exclusive && entry.mode == LockMode::Shared {
                // アップグレード: 他の保持者がいないか確認
                if entry.holders.len() == 1 {
                    entry.mode = LockMode::Exclusive;
                    return Ok(());
                }
                return Err(TxnError::LockConflict);
            }
            return Ok(());
        }

        // 互換性チェック
        if entry.mode == LockMode::Shared && mode == LockMode::Shared {
            // 共有-共有は互換
            entry.holders.insert(txn_id);
            self.held_locks
                .entry(txn_id)
                .or_default()
                .insert(key.to_string());
            return Ok(());
        }

        // 排他ロック競合
        Err(TxnError::LockConflict)
    }

    /// ロックを解放。
    pub fn release(&mut self, txn_id: TxnId, key: &str) -> bool {
        if let Some(entry) = self.locks.get_mut(key) {
            if entry.holders.remove(&txn_id) {
                if let Some(held) = self.held_locks.get_mut(&txn_id) {
                    held.remove(key);
                }
                // 待機者がいれば昇格
                if entry.holders.is_empty() {
                    if let Some((waiter_txn, waiter_mode)) = entry.waiters.pop_front() {
                        entry.holders.insert(waiter_txn);
                        entry.mode = waiter_mode;
                        self.held_locks
                            .entry(waiter_txn)
                            .or_default()
                            .insert(key.to_string());
                    }
                }
                return true;
            }
        }
        false
    }

    /// トランザクションの全ロックを解放。
    pub fn release_all(&mut self, txn_id: TxnId) -> usize {
        let keys: Vec<String> = self
            .held_locks
            .get(&txn_id)
            .map_or_else(Vec::new, |ks| ks.iter().cloned().collect());

        let count = keys.len();
        for key in &keys {
            self.release(txn_id, key);
        }
        self.held_locks.remove(&txn_id);
        count
    }

    /// デッドロック検出 (wait-for グラフの循環検出)。
    #[must_use]
    pub fn detect_deadlock(&self) -> Option<Vec<TxnId>> {
        // wait-for グラフを構築
        //
        // BTreeMap/BTreeSet を使う (HashMap/HashSet ではない): このグラフは直後に
        // キー順 (`.keys()`) とネイバー順 (`for &next in neighbors`) で DFS 走査され、
        // DFS の開始ノード順・探索順が「どの循環を検出するか」「返す Vec<TxnId> の
        // 回転順」という外部可視な戻り値そのものを左右する。HashMap/HashSet は
        // `RandomState` によりプロセス内でもインスタンスごとに反復順が変わりうるため、
        // 同一のロック状態に対して detect_deadlock() を呼ぶたびに異なる循環 / 異なる
        // victim が返り得る (非決定性)。BTreeMap/BTreeSet はキー順序 (TxnId = u64 の
        // 全順序) で一意に反復順が定まるため、この非決定性を排除する。
        let mut wait_for: BTreeMap<TxnId, BTreeSet<TxnId>> = BTreeMap::new();

        for entry in self.locks.values() {
            for &(waiter, _) in &entry.waiters {
                for &holder in &entry.holders {
                    wait_for.entry(waiter).or_default().insert(holder);
                }
            }
        }

        // DFS で循環検出
        let mut visited = BTreeSet::new();
        let mut in_stack = BTreeSet::new();
        let mut path = Vec::new();

        for &txn in wait_for.keys() {
            if !visited.contains(&txn) {
                if let Some(cycle) =
                    Self::dfs_cycle(&wait_for, txn, &mut visited, &mut in_stack, &mut path)
                {
                    return Some(cycle);
                }
            }
        }

        None
    }

    /// DFS 循環検出ヘルパー。
    fn dfs_cycle(
        graph: &BTreeMap<TxnId, BTreeSet<TxnId>>,
        node: TxnId,
        visited: &mut BTreeSet<TxnId>,
        in_stack: &mut BTreeSet<TxnId>,
        path: &mut Vec<TxnId>,
    ) -> Option<Vec<TxnId>> {
        visited.insert(node);
        in_stack.insert(node);
        path.push(node);

        if let Some(neighbors) = graph.get(&node) {
            for &next in neighbors {
                if !visited.contains(&next) {
                    if let Some(cycle) = Self::dfs_cycle(graph, next, visited, in_stack, path) {
                        return Some(cycle);
                    }
                } else if in_stack.contains(&next) {
                    // 循環検出
                    let start = path.iter().position(|&n| n == next).unwrap_or(0);
                    return Some(path[start..].to_vec());
                }
            }
        }

        path.pop();
        in_stack.remove(&node);
        None
    }

    /// 保持ロック数。
    #[must_use]
    pub fn lock_count(&self) -> usize {
        self.locks
            .values()
            .filter(|e| !e.holders.is_empty())
            .count()
    }
}

/// トランザクションエラー。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TxnError {
    /// トランザクションがアクティブでない。
    NotActive,
    /// 書き込み競合。
    WriteConflict,
    /// ロック競合。
    LockConflict,
    /// デッドロック検出。
    Deadlock,
}

impl std::fmt::Display for TxnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotActive => write!(f, "Transaction not active"),
            Self::WriteConflict => write!(f, "Write-write conflict"),
            Self::LockConflict => write!(f, "Lock conflict"),
            Self::Deadlock => write!(f, "Deadlock detected"),
        }
    }
}

impl std::error::Error for TxnError {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mvcc_begin_commit() {
        let mut store: MvccStore<f32> = MvccStore::new();
        let txn = store.begin();
        store.register(txn);
        assert_eq!(store.status(txn), Some(TxnStatus::Active));

        let ts = store.commit(txn).unwrap();
        assert!(ts > 0);
        assert_eq!(store.status(txn), Some(TxnStatus::Committed));
    }

    #[test]
    fn mvcc_write_read() {
        let mut store: MvccStore<f32> = MvccStore::new();
        let txn = store.begin();
        store.register(txn);

        store.write(txn, "key1".to_string(), 42.0).unwrap();
        let val = store.read(txn, "key1");
        assert_eq!(val, Some(42.0));

        store.commit(txn).unwrap();
    }

    #[test]
    fn mvcc_snapshot_isolation() {
        let mut store: MvccStore<f32> = MvccStore::new();

        // T1 が書き込み + コミット
        let t1 = store.begin();
        store.register(t1);
        store.write(t1, "key".to_string(), 10.0).unwrap();
        store.commit(t1).unwrap();

        // T2 開始 (T1 コミット後)
        let t2 = store.begin();
        store.register(t2);

        // T3 が書き込み (T2 開始後)
        let t3 = store.begin();
        store.register(t3);
        store.write(t3, "key".to_string(), 20.0).unwrap();
        store.commit(t3).unwrap();

        // T2 は T1 の値 (10.0) を見る (スナップショット分離)
        let val = store.read(t2, "key");
        assert_eq!(val, Some(10.0));
    }

    #[test]
    fn mvcc_abort_removes_writes() {
        let mut store: MvccStore<f32> = MvccStore::new();
        let txn = store.begin();
        store.register(txn);
        store.write(txn, "key".to_string(), 99.0).unwrap();
        store.abort(txn).unwrap();

        assert_eq!(store.status(txn), Some(TxnStatus::Aborted));

        // 新しいトランザクションからは見えない
        let t2 = store.begin();
        store.register(t2);
        assert!(store.read(t2, "key").is_none());
    }

    #[test]
    fn mvcc_write_conflict() {
        let mut store: MvccStore<f32> = MvccStore::new();
        let t1 = store.begin();
        store.register(t1);
        let t2 = store.begin();
        store.register(t2);

        store.write(t1, "key".to_string(), 1.0).unwrap();
        let result = store.write(t2, "key".to_string(), 2.0);
        assert_eq!(result, Err(TxnError::WriteConflict));
    }

    #[test]
    fn mvcc_gc() {
        let mut store: MvccStore<f32> = MvccStore::new();

        for i in 0..5 {
            let txn = store.begin();
            store.register(txn);
            store.write(txn, "key".to_string(), i as f32).unwrap_or(()); // 競合は無視
            store.commit(txn).unwrap_or(0);
        }

        let removed = store.gc(100);
        assert!(removed > 0);
    }

    #[test]
    fn mvcc_default() {
        let store: MvccStore<f32> = MvccStore::default();
        assert_eq!(store.lock_count_internal(), 0);
    }

    #[test]
    fn lock_acquire_shared() {
        let mut lm = LockManager::new();
        assert!(lm.acquire(1, "key", LockMode::Shared).is_ok());
        assert!(lm.acquire(2, "key", LockMode::Shared).is_ok());
        assert_eq!(lm.lock_count(), 1);
    }

    #[test]
    fn lock_exclusive_conflict() {
        let mut lm = LockManager::new();
        lm.acquire(1, "key", LockMode::Exclusive).unwrap();
        let result = lm.acquire(2, "key", LockMode::Shared);
        assert_eq!(result, Err(TxnError::LockConflict));
    }

    #[test]
    fn lock_release_all() {
        let mut lm = LockManager::new();
        lm.acquire(1, "a", LockMode::Shared).unwrap();
        lm.acquire(1, "b", LockMode::Exclusive).unwrap();
        let released = lm.release_all(1);
        assert_eq!(released, 2);
        assert_eq!(lm.lock_count(), 0);
    }

    #[test]
    fn lock_upgrade() {
        let mut lm = LockManager::new();
        lm.acquire(1, "key", LockMode::Shared).unwrap();
        // 単独保持者は排他にアップグレード可能
        assert!(lm.acquire(1, "key", LockMode::Exclusive).is_ok());
    }

    #[test]
    fn lock_manager_default() {
        let lm = LockManager::default();
        assert_eq!(lm.lock_count(), 0);
    }

    #[test]
    fn deadlock_detection_no_cycle() {
        let lm = LockManager::new();
        assert!(lm.detect_deadlock().is_none());
    }

    /// `LockEntry`/`LockManager` の private field を直接組み立てて wait-for グラフを
    /// 作る helper。
    ///
    /// `acquire()` は現状どの経路でも `LockEntry::waiters` に push しない (`waiters`
    /// への書き込みは crate 全体を探しても存在せず、`pop_front` される側でしか現れ
    /// ない) ので、公開 API だけでは `detect_deadlock()` が実際に循環を見つける状態
    /// を作れない。これは G13 (本テストの対象、`HashMap`/`HashSet` の反復順依存) とは別の
    /// 配線上の欠落であり、本 commit の scope 外として直さず、private field への
    /// 直接代入でテスト用の wait-for グラフを構築する (このテストモジュールは
    /// `super::*` を介して同一ファイル内の private field にアクセスできる)。
    ///
    /// `key -> (holder, waiter)` の対応で 2 本の独立した 2-cycle
    /// (txn 1 <-> txn 2、txn 3 <-> txn 4) を作る。`key_order` は `LockManager::locks`
    /// (`HashMap`) へ挿入する順序を変えるためのもの。
    fn build_two_cycle_lock_manager(key_order: &[&str]) -> LockManager {
        let mut lm = LockManager::new();
        for &k in key_order {
            let (holder, waiter): (TxnId, TxnId) = match k {
                "a" => (1, 2), // txn 2 が txn 1 の保持するロックを待つ
                "b" => (2, 1), // txn 1 が txn 2 の保持するロックを待つ (a と合わせて 1<->2 の循環)
                "c" => (3, 4), // txn 4 が txn 3 の保持するロックを待つ
                "d" => (4, 3), // txn 3 が txn 4 の保持するロックを待つ (c と合わせて 3<->4 の循環)
                _ => unreachable!("unknown key in test fixture: {k}"),
            };
            let mut holders = HashSet::new();
            holders.insert(holder);
            lm.locks.insert(
                k.to_string(),
                LockEntry {
                    holders,
                    mode: LockMode::Exclusive,
                    waiters: VecDeque::from(vec![(waiter, LockMode::Exclusive)]),
                },
            );
        }
        lm
    }

    /// G13: `detect_deadlock()` は同一のロック状態に対して常に同じ循環を返す。
    ///
    /// 2 つの独立した 2-cycle (1<->2、3<->4) がある状態では、
    /// `detect_deadlock()` が最初に見つける循環は wait-for グラフの反復順に左右さ
    /// れる。`detect_deadlock()` は呼ばれるたびに内部で新しい wait-for グラフを
    /// 構築するため、`wait_for`/`visited`/`in_stack` が `HashMap`/`HashSet` だった旧実装
    /// では、`RandomState` がプロセス内でもインスタンスごとに反復順を変えるせいで、
    /// 「同じ `&self` 状態に対する呼び出しのたびに」異なる循環が返り得た。
    /// `BTreeMap`/`BTreeSet` 化した現在の実装は `TxnId` (= `u64`) の全順序でキー・ネイバー
    /// の反復順が一意に定まるため、何度呼んでも結果は同じでなければならない。
    #[test]
    fn detect_deadlock_result_is_order_independent() {
        let lm = build_two_cycle_lock_manager(&["a", "b", "c", "d"]);
        let first = lm.detect_deadlock();

        // 同一の `&self` 状態に対して detect_deadlock() を繰り返し呼ぶ。各呼び出しは
        // 内部で新しい wait_for グラフを fresh 構築するため、HashMap/HashSet のまま
        // であればここで反復順の揺れが表面化しうる。
        for i in 0..200 {
            let again = lm.detect_deadlock();
            assert_eq!(
                again, first,
                "detect_deadlock() returned a different cycle on repeated call #{i} \
                 against the exact same lock state: first={first:?} now={again:?}"
            );
        }
    }

    /// G13: `detect_deadlock()` の結果は `locks` (`HashMap`) への挿入順序にも依らない。
    ///
    /// `locks` 自体は iteration 結果が wait-for グラフの「内容」(集合としてのメンバ
    /// シップ) にしか効かず、順序には効かない (= 集合への insert は可換) ので、この
    /// assertion は挿入順序を変えても自明に成立しうる。しかし挙動の契約として明示
    /// しておく価値があるため、ここでも固定する。
    #[test]
    fn detect_deadlock_result_is_insertion_order_independent() {
        let orders: [&[&str]; 4] = [
            &["a", "b", "c", "d"],
            &["d", "c", "b", "a"],
            &["b", "d", "a", "c"],
            &["c", "a", "d", "b"],
        ];

        let results: Vec<Option<Vec<TxnId>>> = orders
            .iter()
            .map(|order| build_two_cycle_lock_manager(order).detect_deadlock())
            .collect();

        for (order, result) in orders.iter().zip(results.iter()) {
            assert_eq!(
                *result, results[0],
                "insertion order {order:?} produced a different cycle than {:?}: {result:?}",
                orders[0]
            );
        }
    }

    #[test]
    fn txn_error_display() {
        assert_eq!(TxnError::NotActive.to_string(), "Transaction not active");
        assert_eq!(TxnError::WriteConflict.to_string(), "Write-write conflict");
        assert_eq!(TxnError::LockConflict.to_string(), "Lock conflict");
        assert_eq!(TxnError::Deadlock.to_string(), "Deadlock detected");
    }

    #[test]
    fn lock_mode_display() {
        assert_eq!(LockMode::Shared.to_string(), "Shared");
        assert_eq!(LockMode::Exclusive.to_string(), "Exclusive");
    }
}
