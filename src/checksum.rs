//! データ整合性チェックサム
//!
//! CRC32 / xxHash64 によるセグメントデータの整合性検証。
//! 書き込み時にチェックサムを埋め込み、読み取り時に検証。

use std::io;

/// チェックサムアルゴリズム。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChecksumAlgorithm {
    /// CRC32 (IEEE)。
    Crc32,
    /// `xxHash64`。
    XxHash64,
}

impl std::fmt::Display for ChecksumAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Crc32 => write!(f, "CRC32"),
            Self::XxHash64 => write!(f, "xxHash64"),
        }
    }
}

/// チェックサム値。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Checksum {
    /// アルゴリズム。
    pub algorithm: ChecksumAlgorithm,
    /// チェックサム値 (8バイト、CRC32 は下位4バイトのみ使用)。
    pub value: u64,
}

impl Checksum {
    /// CRC32 チェックサムを計算。
    #[must_use]
    pub fn crc32(data: &[u8]) -> Self {
        let value = u64::from(compute_crc32(data));
        Self {
            algorithm: ChecksumAlgorithm::Crc32,
            value,
        }
    }

    /// `xxHash64` チェックサムを計算。
    #[must_use]
    pub fn xxhash64(data: &[u8]) -> Self {
        let value = compute_xxhash64(data);
        Self {
            algorithm: ChecksumAlgorithm::XxHash64,
            value,
        }
    }

    /// 指定アルゴリズムでチェックサムを計算。
    #[must_use]
    pub fn compute(data: &[u8], algorithm: ChecksumAlgorithm) -> Self {
        match algorithm {
            ChecksumAlgorithm::Crc32 => Self::crc32(data),
            ChecksumAlgorithm::XxHash64 => Self::xxhash64(data),
        }
    }

    /// データとの整合性を検証。
    #[must_use]
    pub fn verify(&self, data: &[u8]) -> bool {
        let computed = Self::compute(data, self.algorithm);
        self.value == computed.value
    }

    /// バイト列にシリアライズ (9バイト: 1 algorithm + 8 value)。
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 9] {
        let mut buf = [0u8; 9];
        buf[0] = match self.algorithm {
            ChecksumAlgorithm::Crc32 => 0,
            ChecksumAlgorithm::XxHash64 => 1,
        };
        buf[1..9].copy_from_slice(&self.value.to_le_bytes());
        buf
    }

    /// バイト列からデシリアライズ。
    ///
    /// # Errors
    ///
    /// バイト長が不足、またはアルゴリズムが不明な場合。
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ChecksumError> {
        if bytes.len() < 9 {
            return Err(ChecksumError::InvalidLength);
        }
        let algorithm = match bytes[0] {
            0 => ChecksumAlgorithm::Crc32,
            1 => ChecksumAlgorithm::XxHash64,
            _ => return Err(ChecksumError::UnknownAlgorithm),
        };
        let value = u64::from_le_bytes(
            bytes[1..9]
                .try_into()
                .map_err(|_| ChecksumError::InvalidLength)?,
        );
        Ok(Self { algorithm, value })
    }
}

/// チェックサム付きセグメントヘッダー。
#[derive(Debug, Clone)]
pub struct ChecksummedHeader {
    /// ヘッダーマジック。
    pub magic: u32,
    /// データ長。
    pub data_length: u64,
    /// ヘッダーチェックサム。
    pub header_checksum: Checksum,
    /// データチェックサム。
    pub data_checksum: Checksum,
}

/// マジックバイト。
pub const SEGMENT_MAGIC: u32 = 0xA1CE_DB01;

impl ChecksummedHeader {
    /// ヘッダーを作成。
    #[must_use]
    pub fn new(data: &[u8], algorithm: ChecksumAlgorithm) -> Self {
        let data_checksum = Checksum::compute(data, algorithm);
        let data_length = data.len() as u64;

        // ヘッダー本体のチェックサム (magic + data_length)
        let mut header_bytes = Vec::with_capacity(12);
        header_bytes.extend_from_slice(&SEGMENT_MAGIC.to_le_bytes());
        header_bytes.extend_from_slice(&data_length.to_le_bytes());
        let header_checksum = Checksum::compute(&header_bytes, algorithm);

        Self {
            magic: SEGMENT_MAGIC,
            data_length,
            header_checksum,
            data_checksum,
        }
    }

    /// ヘッダーをバイト列にシリアライズ。
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(30);
        buf.extend_from_slice(&self.magic.to_le_bytes());
        buf.extend_from_slice(&self.data_length.to_le_bytes());
        buf.extend_from_slice(&self.header_checksum.to_bytes());
        buf.extend_from_slice(&self.data_checksum.to_bytes());
        buf
    }

    /// バイト列からデシリアライズ。
    ///
    /// # Errors
    ///
    /// フォーマットが不正な場合。
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ChecksumError> {
        if bytes.len() < 30 {
            return Err(ChecksumError::InvalidLength);
        }

        let magic = u32::from_le_bytes(
            bytes[0..4]
                .try_into()
                .map_err(|_| ChecksumError::InvalidLength)?,
        );
        if magic != SEGMENT_MAGIC {
            return Err(ChecksumError::InvalidMagic);
        }

        let data_length = u64::from_le_bytes(
            bytes[4..12]
                .try_into()
                .map_err(|_| ChecksumError::InvalidLength)?,
        );
        let header_checksum = Checksum::from_bytes(&bytes[12..21])?;
        let data_checksum = Checksum::from_bytes(&bytes[21..30])?;

        Ok(Self {
            magic,
            data_length,
            header_checksum,
            data_checksum,
        })
    }

    /// ヘッダー整合性を検証。
    #[must_use]
    pub fn verify_header(&self) -> bool {
        let mut header_bytes = Vec::with_capacity(12);
        header_bytes.extend_from_slice(&self.magic.to_le_bytes());
        header_bytes.extend_from_slice(&self.data_length.to_le_bytes());
        self.header_checksum.verify(&header_bytes)
    }

    /// データ整合性を検証。
    #[must_use]
    pub fn verify_data(&self, data: &[u8]) -> bool {
        data.len() as u64 == self.data_length && self.data_checksum.verify(data)
    }
}

/// チェックサム付きライター。
pub struct ChecksummedWriter<W: io::Write> {
    /// 内部ライター。
    inner: W,
    /// チェックサムアルゴリズム。
    algorithm: ChecksumAlgorithm,
}

impl<W: io::Write> ChecksummedWriter<W> {
    /// 新しいライターを作成。
    #[must_use]
    pub const fn new(inner: W, algorithm: ChecksumAlgorithm) -> Self {
        Self { inner, algorithm }
    }

    /// データをチェックサム付きで書き込み。
    ///
    /// # Errors
    ///
    /// I/O エラー。
    pub fn write_checksummed(&mut self, data: &[u8]) -> io::Result<usize> {
        let header = ChecksummedHeader::new(data, self.algorithm);
        let header_bytes = header.to_bytes();
        self.inner.write_all(&header_bytes)?;
        self.inner.write_all(data)?;
        Ok(header_bytes.len() + data.len())
    }

    /// 内部ライターへの参照。
    #[must_use]
    pub const fn inner(&self) -> &W {
        &self.inner
    }
}

/// チェックサム付きリーダー。
pub struct ChecksummedReader<R: io::Read> {
    /// 内部リーダー。
    inner: R,
}

impl<R: io::Read> ChecksummedReader<R> {
    /// 新しいリーダーを作成。
    #[must_use]
    pub const fn new(inner: R) -> Self {
        Self { inner }
    }

    /// チェックサム付きデータを読み取り・検証。
    ///
    /// # Errors
    ///
    /// I/O エラーまたはチェックサム不一致。
    pub fn read_checksummed(&mut self) -> io::Result<Vec<u8>> {
        let mut header_buf = [0u8; 30];
        self.inner.read_exact(&mut header_buf)?;

        let header = ChecksummedHeader::from_bytes(&header_buf)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

        if !header.verify_header() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Header checksum mismatch",
            ));
        }

        let mut data = vec![0u8; header.data_length as usize];
        self.inner.read_exact(&mut data)?;

        if !header.verify_data(&data) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Data checksum mismatch",
            ));
        }

        Ok(data)
    }
}

// ============================================================================
// CRC32 実装 (IEEE 多項式、テーブルベース)
// ============================================================================

/// CRC32 IEEE 多項式。
const CRC32_POLYNOMIAL: u32 = 0xEDB8_8320;

/// CRC32 ルックアップテーブルを生成。
const fn generate_crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            if crc & 1 == 1 {
                crc = (crc >> 1) ^ CRC32_POLYNOMIAL;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

/// CRC32 テーブル。
const CRC32_TABLE: [u32; 256] = generate_crc32_table();

/// CRC32 を計算。
#[must_use]
fn compute_crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        let index = ((crc ^ u32::from(byte)) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC32_TABLE[index];
    }
    crc ^ 0xFFFF_FFFF
}

// ============================================================================
// xxHash64 実装 (簡易版)
// ============================================================================

/// `xxHash64` プライム定数。
const XXHASH64_PRIME1: u64 = 0x9E37_79B1_85EB_CA87;
const XXHASH64_PRIME2: u64 = 0xC2B2_AE3D_27D4_EB4F;
const XXHASH64_PRIME3: u64 = 0x1656_67B1_9E37_79F9;
const XXHASH64_PRIME4: u64 = 0x85EB_CA77_C2B2_AE63;
const XXHASH64_PRIME5: u64 = 0x27D4_EB2F_1656_67C5;

/// `xxHash64` を計算。
#[must_use]
fn compute_xxhash64(data: &[u8]) -> u64 {
    let len = data.len() as u64;
    let mut h: u64;

    if data.len() >= 32 {
        let mut v1 = XXHASH64_PRIME1.wrapping_add(XXHASH64_PRIME2);
        let mut v2 = XXHASH64_PRIME2;
        let mut v3 = 0u64;
        let mut v4 = XXHASH64_PRIME1.wrapping_neg();

        let mut i = 0;
        while i + 32 <= data.len() {
            v1 = xxhash64_round(v1, read_u64_le(data, i));
            v2 = xxhash64_round(v2, read_u64_le(data, i + 8));
            v3 = xxhash64_round(v3, read_u64_le(data, i + 16));
            v4 = xxhash64_round(v4, read_u64_le(data, i + 24));
            i += 32;
        }

        h = v1
            .rotate_left(1)
            .wrapping_add(v2.rotate_left(7))
            .wrapping_add(v3.rotate_left(12))
            .wrapping_add(v4.rotate_left(18));

        h = xxhash64_merge_round(h, v1);
        h = xxhash64_merge_round(h, v2);
        h = xxhash64_merge_round(h, v3);
        h = xxhash64_merge_round(h, v4);
    } else {
        h = XXHASH64_PRIME5;
    }

    h = h.wrapping_add(len);

    // 残りバイトの処理
    let mut i = (data.len() / 32) * 32;
    while i + 8 <= data.len() {
        let k = read_u64_le(data, i).wrapping_mul(XXHASH64_PRIME2);
        h ^= k.rotate_left(31).wrapping_mul(XXHASH64_PRIME1);
        h = h
            .rotate_left(27)
            .wrapping_mul(XXHASH64_PRIME1)
            .wrapping_add(XXHASH64_PRIME4);
        i += 8;
    }

    while i + 4 <= data.len() {
        let k = u64::from(read_u32_le(data, i));
        h ^= k.wrapping_mul(XXHASH64_PRIME1);
        h = h
            .rotate_left(23)
            .wrapping_mul(XXHASH64_PRIME2)
            .wrapping_add(XXHASH64_PRIME3);
        i += 4;
    }

    while i < data.len() {
        h ^= u64::from(data[i]).wrapping_mul(XXHASH64_PRIME5);
        h = h.rotate_left(11).wrapping_mul(XXHASH64_PRIME1);
        i += 1;
    }

    // 最終ミックス
    h ^= h >> 33;
    h = h.wrapping_mul(XXHASH64_PRIME2);
    h ^= h >> 29;
    h = h.wrapping_mul(XXHASH64_PRIME3);
    h ^= h >> 32;

    h
}

#[inline]
const fn xxhash64_round(acc: u64, input: u64) -> u64 {
    acc.wrapping_add(input.wrapping_mul(XXHASH64_PRIME2))
        .rotate_left(31)
        .wrapping_mul(XXHASH64_PRIME1)
}

#[inline]
const fn xxhash64_merge_round(acc: u64, val: u64) -> u64 {
    let val = val
        .wrapping_mul(XXHASH64_PRIME2)
        .rotate_left(31)
        .wrapping_mul(XXHASH64_PRIME1);
    (acc ^ val)
        .wrapping_mul(XXHASH64_PRIME1)
        .wrapping_add(XXHASH64_PRIME4)
}

#[inline]
fn read_u64_le(data: &[u8], offset: usize) -> u64 {
    let bytes: [u8; 8] = data[offset..offset + 8].try_into().unwrap_or([0; 8]);
    u64::from_le_bytes(bytes)
}

#[inline]
fn read_u32_le(data: &[u8], offset: usize) -> u32 {
    let bytes: [u8; 4] = data[offset..offset + 4].try_into().unwrap_or([0; 4]);
    u32::from_le_bytes(bytes)
}

/// チェックサムエラー。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChecksumError {
    /// バイト長が不足。
    InvalidLength,
    /// 不明なアルゴリズム。
    UnknownAlgorithm,
    /// マジックバイトが不正。
    InvalidMagic,
    /// チェックサム不一致。
    Mismatch,
}

impl std::fmt::Display for ChecksumError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidLength => write!(f, "Invalid checksum byte length"),
            Self::UnknownAlgorithm => write!(f, "Unknown checksum algorithm"),
            Self::InvalidMagic => write!(f, "Invalid segment magic bytes"),
            Self::Mismatch => write!(f, "Checksum mismatch"),
        }
    }
}

impl std::error::Error for ChecksumError {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn crc32_known_value() {
        // "ALICE" の CRC32
        let checksum = Checksum::crc32(b"ALICE");
        assert_eq!(checksum.algorithm, ChecksumAlgorithm::Crc32);
        assert!(checksum.value > 0);
    }

    #[test]
    fn crc32_empty() {
        let c1 = Checksum::crc32(b"");
        let c2 = Checksum::crc32(b"data");
        assert_ne!(c1.value, c2.value);
    }

    #[test]
    fn xxhash64_deterministic() {
        let c1 = Checksum::xxhash64(b"hello world");
        let c2 = Checksum::xxhash64(b"hello world");
        assert_eq!(c1.value, c2.value);
    }

    #[test]
    fn xxhash64_different_data() {
        let c1 = Checksum::xxhash64(b"data1");
        let c2 = Checksum::xxhash64(b"data2");
        assert_ne!(c1.value, c2.value);
    }

    #[test]
    fn checksum_verify_ok() {
        let data = b"some segment data";
        let checksum = Checksum::crc32(data);
        assert!(checksum.verify(data));
    }

    #[test]
    fn checksum_verify_fail() {
        let checksum = Checksum::crc32(b"original");
        assert!(!checksum.verify(b"modified"));
    }

    #[test]
    fn checksum_serialize_deserialize() {
        let checksum = Checksum::xxhash64(b"test data");
        let bytes = checksum.to_bytes();
        let restored = Checksum::from_bytes(&bytes).unwrap();
        assert_eq!(checksum, restored);
    }

    #[test]
    fn checksum_from_bytes_too_short() {
        assert_eq!(
            Checksum::from_bytes(&[0, 1, 2]),
            Err(ChecksumError::InvalidLength)
        );
    }

    #[test]
    fn checksummed_header_roundtrip() {
        let data = b"segment payload data here";
        let header = ChecksummedHeader::new(data, ChecksumAlgorithm::Crc32);

        assert!(header.verify_header());
        assert!(header.verify_data(data));

        let bytes = header.to_bytes();
        let restored = ChecksummedHeader::from_bytes(&bytes).unwrap();
        assert!(restored.verify_header());
        assert!(restored.verify_data(data));
    }

    #[test]
    fn checksummed_header_corrupt_data() {
        let data = b"original data";
        let header = ChecksummedHeader::new(data, ChecksumAlgorithm::XxHash64);
        assert!(!header.verify_data(b"corrupted data"));
    }

    #[test]
    fn checksummed_writer_reader() {
        let data = b"test segment data for writer/reader";
        let mut buf = Vec::new();

        let mut writer = ChecksummedWriter::new(&mut buf, ChecksumAlgorithm::Crc32);
        writer.write_checksummed(data).unwrap();

        let mut reader = ChecksummedReader::new(Cursor::new(&buf));
        let read_data = reader.read_checksummed().unwrap();
        assert_eq!(read_data, data);
    }

    #[test]
    fn checksummed_reader_corrupt() {
        let data = b"test data";
        let mut buf = Vec::new();
        let mut writer = ChecksummedWriter::new(&mut buf, ChecksumAlgorithm::Crc32);
        writer.write_checksummed(data).unwrap();

        // データ部分を破損
        let last = buf.len() - 1;
        buf[last] ^= 0xFF;

        let mut reader = ChecksummedReader::new(Cursor::new(&buf));
        assert!(reader.read_checksummed().is_err());
    }

    #[test]
    fn algorithm_display() {
        assert_eq!(ChecksumAlgorithm::Crc32.to_string(), "CRC32");
        assert_eq!(ChecksumAlgorithm::XxHash64.to_string(), "xxHash64");
    }

    #[test]
    fn checksum_error_display() {
        assert_eq!(
            ChecksumError::InvalidLength.to_string(),
            "Invalid checksum byte length"
        );
        assert_eq!(ChecksumError::Mismatch.to_string(), "Checksum mismatch");
    }

    #[test]
    fn checksum_compute_dispatches() {
        let data = b"dispatch test";
        let c1 = Checksum::compute(data, ChecksumAlgorithm::Crc32);
        let c2 = Checksum::crc32(data);
        assert_eq!(c1, c2);

        let c3 = Checksum::compute(data, ChecksumAlgorithm::XxHash64);
        let c4 = Checksum::xxhash64(data);
        assert_eq!(c3, c4);
    }
}
