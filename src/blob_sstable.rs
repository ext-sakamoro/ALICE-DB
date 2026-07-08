//! Sorted, immutable `SSTable` for the blob key-value store (v0.2.0-alpha.4).
//!
//! Every [`crate::blob::BlobStorage`] instance may own at most one
//! `BlobSstable` file (`blob.sst` sitting next to `blob.wal`). It holds
//! the "settled" portion of the store — every live key at the moment of
//! the most recent flush. The still-mutable diff on top of it lives in
//! the WAL.
//!
//! # Read path relationship
//!
//! ```text
//!   on-disk state = blob.sst  (immutable, sorted, no tombstones)
//!                 + blob.wal  (append-only diff since the flush)
//!
//!   in-memory state = load(blob.sst) then replay(blob.wal)
//! ```
//!
//! A `BlobSstable` written by the current version *never* stores
//! tombstones: those exist only as an intermediate state in the WAL and
//! are physically removed by the flush that produced this file. This
//! means an `SSTable` read at any time is a truthful, dense snapshot of
//! live keys.
//!
//! # File layout
//!
//! ```text
//! Header (24 bytes):
//!   0   8   magic          "ALICEBBS"
//!   8   4   version        u32 LE (currently 1)
//!  12   8   num_records    u64 LE (may be zero for an empty snapshot)
//!  20   4   reserved       u32 LE (must be zero — v0.2.0-alpha.4 slot)
//!
//! Records section (sorted ascending by key):
//!   [record 1][record 2]...[record N]
//!
//! Each record:
//!   0   4   key_len        u32 LE
//!   4   4   value_len      u32 LE
//!   8   1   value_kind     u8  (0x00 = Raw, 0x01 = Compressed)
//!   9   key_len            key bytes
//!   9+key_len   value_len  value bytes
//!   trailer  4              crc32c over the entire record above
//!
//! Footer (16 bytes, at end of file):
//!   ...  8   records_size   u64 LE (total byte length of the records section)
//!   ...  8   magic          "ALICEEND"
//! ```
//!
//! # Atomic replacement
//!
//! `write_from_iter` writes to a sibling `.tmp` path and then renames.
//! On the platforms we support (Linux, macOS) `rename(2)` is atomic
//! across the same filesystem, so a reader never observes a partially
//! written `SSTable`.

use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::blob::BlobValue;

/// File-format magic marking the start of a blob `SSTable`.
const HEADER_MAGIC: &[u8; 8] = b"ALICEBBS";
/// File-format magic marking the end of a blob `SSTable`.
const FOOTER_MAGIC: &[u8; 8] = b"ALICEEND";
/// Format version. Every write of the current codebase stamps this;
/// readers reject anything else.
const CURRENT_VERSION: u32 = 1;
/// Bytes occupied by the fixed-size header.
const HEADER_LEN: usize = 24;
/// Bytes occupied by the fixed-size footer.
const FOOTER_LEN: usize = 16;
/// Encoded flag for `BlobValue::Raw`.
const VALUE_KIND_RAW: u8 = 0x00;
/// Encoded flag for `BlobValue::Compressed`.
const VALUE_KIND_COMPRESSED: u8 = 0x01;

/// Number of decimal digits in the zero-padded sequence part of an
/// `Append`-mode `SSTable` filename (`blob-{seq:06}.sst`).
const SSTABLE_SEQ_WIDTH: usize = 6;

/// How a flush should treat the existing `SSTable` files in the store.
///
/// Introduced in v0.2.0-alpha.5.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FlushMode {
    /// Every flush rewrites the single canonical `blob.sst`. Matches
    /// v0.2.0-alpha.4 behaviour and remains the default so pre-α-3.3
    /// databases and callers observe no change.
    #[default]
    Overwrite,
    /// Every flush produces a new sequentially numbered `SSTable` file
    /// (`blob-000001.sst`, `blob-000002.sst`, …). Older `SSTables` are
    /// left in place until [`crate::blob::BlobStorage::compact_all_sstables`]
    /// (or the auto-compaction threshold on
    /// [`crate::blob::BlobStorageConfig`]) merges them back into one.
    /// This makes each flush O(delta) rather than O(N), at the cost of
    /// eventually running a full merge.
    Append,
}

/// Return the sequence number encoded in a blob `SSTable` filename, if
/// the filename matches the expected format.
///
/// Accepted forms:
/// - `blob.sst` — treated as sequence `0` for backward compatibility
///   with v0.2.0-alpha.2 through v0.2.0-alpha.4 stores.
/// - `blob-{seq:06}.sst` — the α-3.3 append-mode form; `seq` may be any
///   decimal value that fits in a `u64`.
///
/// Any other filename (temp files, unrelated `.sst`, etc.) yields
/// `None` so the caller can skip it.
#[must_use]
pub fn parse_sstable_seq(name: &str) -> Option<u64> {
    if name == "blob.sst" {
        return Some(0);
    }
    let rest = name.strip_prefix("blob-")?;
    let digits = rest.strip_suffix(".sst")?;
    if digits.len() < SSTABLE_SEQ_WIDTH || !digits.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    digits.parse::<u64>().ok()
}

/// Generate the filename for a new append-mode `SSTable` with the given
/// sequence number, zero-padded to [`SSTABLE_SEQ_WIDTH`] digits.
#[must_use]
pub fn sstable_filename_for_seq(seq: u64) -> String {
    let width = SSTABLE_SEQ_WIDTH;
    format!("blob-{seq:0width$}.sst")
}

/// Enumerate every blob `SSTable` currently living in `dir`, sorted by
/// sequence number ascending.
///
/// Non-`SSTable` files are silently skipped. Callers that expect only
/// the α-3.2a single-file layout can either continue to use `blob.sst`
/// directly or opt into `FlushMode::Append` and let this helper walk
/// the directory.
///
/// # Errors
/// Propagates the underlying [`std::fs::read_dir`] error.
pub fn enumerate_sstables(dir: impl AsRef<Path>) -> io::Result<Vec<PathBuf>> {
    let dir = dir.as_ref();
    let mut hits: Vec<(u64, PathBuf)> = Vec::new();
    let read = match std::fs::read_dir(dir) {
        Ok(r) => r,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    for entry in read {
        let entry = entry?;
        let file_name = entry.file_name();
        let Some(name) = file_name.to_str() else {
            continue;
        };
        if let Some(seq) = parse_sstable_seq(name) {
            hits.push((seq, entry.path()));
        }
    }
    hits.sort_by_key(|(seq, _)| *seq);
    Ok(hits.into_iter().map(|(_, p)| p).collect())
}

/// Highest sequence number in use for `SSTables` under `dir`, if any.
///
/// Callers that need the next sequence number for an append-mode flush
/// can add one to this value (saturating at `u64::MAX`).
///
/// # Errors
/// Propagates the underlying [`enumerate_sstables`] error.
pub fn max_sstable_seq(dir: impl AsRef<Path>) -> io::Result<Option<u64>> {
    let hits = enumerate_sstables(dir)?;
    let mut max: Option<u64> = None;
    for path in hits {
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .and_then(parse_sstable_seq);
        if let Some(seq) = name {
            max = Some(match max {
                Some(prev) => prev.max(seq),
                None => seq,
            });
        }
    }
    Ok(max)
}

/// Owned representation of a blob `SSTable`'s on-disk contents after
/// [`BlobSstable::open`].
///
/// Alpha-3.2a loads every record into memory so the in-memory `BTreeMap`
/// can be reconstructed via a single `extend`. A future alpha (α-3.3)
/// will keep the file mmap'd for point lookups without full load.
#[derive(Debug)]
pub struct BlobSstable {
    path: PathBuf,
    records: Vec<(Vec<u8>, BlobValue)>,
}

impl BlobSstable {
    /// The on-disk location of the `SSTable`.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Number of records the `SSTable` holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the `SSTable` contains any records.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Iterate over the records in ascending key order.
    ///
    /// Returned tuples borrow from the `SSTable` so callers avoid cloning
    /// during load. Convert them into owned pairs with `.clone()` or
    /// `.to_vec()` if the `SSTable` will be dropped afterwards.
    pub fn iter(&self) -> impl Iterator<Item = (&[u8], &BlobValue)> {
        self.records.iter().map(|(k, v)| (k.as_slice(), v))
    }

    /// Consume the `SSTable` and return its records as owned tuples.
    ///
    /// Convenient for feeding directly into the [`crate::blob::BlobStorage`]
    /// in-memory `BTreeMap`.
    #[must_use]
    pub fn into_records(self) -> Vec<(Vec<u8>, BlobValue)> {
        self.records
    }

    /// Write a new `SSTable` atomically at `path`.
    ///
    /// `iter` must yield entries in ascending key order and must not
    /// contain [`BlobValue::Tombstone`] — tombstones are a WAL-only
    /// concept and are physically dropped at flush time. The caller is
    /// responsible for filtering them out.
    ///
    /// # Errors
    /// Returns any underlying I/O error. Failed writes leave the sibling
    /// `.tmp` path behind (best-effort cleanup) and never touch the
    /// destination `path`.
    pub fn write_from_iter<'a, I>(path: impl AsRef<Path>, iter: I) -> io::Result<()>
    where
        I: IntoIterator<Item = (&'a [u8], &'a BlobValue)>,
    {
        let path = path.as_ref();
        let tmp_path = tmp_sibling(path);

        // Buffer the records so we can compute `num_records` for the
        // header before writing anything. Alpha-3.2a favours simplicity
        // over streaming; α-3.3 can add a "size-known-in-advance" path.
        let entries: Vec<(&[u8], &BlobValue)> = iter.into_iter().collect();
        let num_records = u64::try_from(entries.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "sstable exceeds u64 record count",
            )
        })?;

        // Reject tombstones early with a diagnostic instead of writing a
        // silently broken file.
        for (key, value) in &entries {
            if matches!(value, BlobValue::Tombstone) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "attempted to write tombstone key {:?} to sstable — tombstones are WAL-only",
                        String::from_utf8_lossy(key)
                    ),
                ));
            }
        }

        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp_path)?;
        let mut writer = BufWriter::new(file);

        // Header
        writer.write_all(HEADER_MAGIC)?;
        writer.write_all(&CURRENT_VERSION.to_le_bytes())?;
        writer.write_all(&num_records.to_le_bytes())?;
        writer.write_all(&0_u32.to_le_bytes())?; // reserved

        // Records
        let mut records_size: u64 = 0;
        for (key, value) in entries {
            let (kind, bytes): (u8, &[u8]) = match value {
                BlobValue::Raw(b) => (VALUE_KIND_RAW, b.as_slice()),
                BlobValue::Compressed(b) => (VALUE_KIND_COMPRESSED, b.as_slice()),
                BlobValue::Tombstone => unreachable!("filtered above"),
            };
            let record = serialise_record(key, kind, bytes)?;
            writer.write_all(&record)?;
            records_size = records_size
                .checked_add(u64::try_from(record.len()).unwrap_or(u64::MAX))
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "records section overflows u64")
                })?;
        }

        // Footer
        writer.write_all(&records_size.to_le_bytes())?;
        writer.write_all(FOOTER_MAGIC)?;

        let file = writer
            .into_inner()
            .map_err(std::io::IntoInnerError::into_error)?;
        file.sync_all()?;
        drop(file);

        std::fs::rename(&tmp_path, path)?;
        Ok(())
    }

    /// Load an `SSTable` from disk. Returns `Ok(None)` if `path` does not
    /// exist, so callers can treat "no `SSTable` yet" and "empty `SSTable`"
    /// as distinct states.
    ///
    /// # Errors
    /// Returns an `io::Error` if the file is present but its header,
    /// footer, or any record CRC is malformed. The file is treated as
    /// corrupt in that case; the caller can decide whether to fall
    /// back to WAL-only load or to abort.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Option<Self>> {
        let path = path.as_ref().to_path_buf();
        let mut file = match File::open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e),
        };

        // Read the whole file — alpha-3.2a is not memory-frugal by design.
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;

        if bytes.len() < HEADER_LEN + FOOTER_LEN {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "sstable is smaller than the fixed header + footer",
            ));
        }
        if &bytes[0..8] != HEADER_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "sstable header magic mismatch",
            ));
        }
        let version = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        if version != CURRENT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported sstable version {version}; expected {CURRENT_VERSION}"),
            ));
        }
        let num_records = u64::from_le_bytes([
            bytes[12], bytes[13], bytes[14], bytes[15], bytes[16], bytes[17], bytes[18], bytes[19],
        ]);

        let footer_start = bytes.len() - FOOTER_LEN;
        if &bytes[footer_start + 8..] != FOOTER_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "sstable footer magic mismatch",
            ));
        }
        let footer_records_size = u64::from_le_bytes([
            bytes[footer_start],
            bytes[footer_start + 1],
            bytes[footer_start + 2],
            bytes[footer_start + 3],
            bytes[footer_start + 4],
            bytes[footer_start + 5],
            bytes[footer_start + 6],
            bytes[footer_start + 7],
        ]);
        let records_slice = &bytes[HEADER_LEN..footer_start];
        if u64::try_from(records_slice.len()).unwrap_or(u64::MAX) != footer_records_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "sstable footer records_size does not match section length",
            ));
        }

        let records = parse_records(records_slice, num_records)?;
        Ok(Some(Self { path, records }))
    }
}

/// Emit the byte encoding of one record plus its trailing CRC32C.
fn serialise_record(key: &[u8], value_kind: u8, value: &[u8]) -> io::Result<Vec<u8>> {
    let key_len = u32::try_from(key.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "sstable key exceeds u32::MAX"))?;
    let value_len = u32::try_from(value.len()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "sstable value exceeds u32::MAX",
        )
    })?;

    // Header (9 bytes) + key + value.
    let mut buf = Vec::with_capacity(9 + key.len() + value.len() + 4);
    buf.extend_from_slice(&key_len.to_le_bytes());
    buf.extend_from_slice(&value_len.to_le_bytes());
    buf.push(value_kind);
    buf.extend_from_slice(key);
    buf.extend_from_slice(value);

    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&buf);
    let crc = hasher.finalize();
    buf.extend_from_slice(&crc.to_le_bytes());
    Ok(buf)
}

/// Parse the records section into `(key, value)` pairs, verifying the
/// per-record CRC as we go.
fn parse_records(bytes: &[u8], expected_count: u64) -> io::Result<Vec<(Vec<u8>, BlobValue)>> {
    let mut out = Vec::with_capacity(usize::try_from(expected_count).unwrap_or(0));
    let mut cursor = 0_usize;
    while cursor < bytes.len() {
        if bytes.len() - cursor < 9 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "sstable record header truncated",
            ));
        }
        let key_len = u32::from_le_bytes([
            bytes[cursor],
            bytes[cursor + 1],
            bytes[cursor + 2],
            bytes[cursor + 3],
        ]) as usize;
        let value_len = u32::from_le_bytes([
            bytes[cursor + 4],
            bytes[cursor + 5],
            bytes[cursor + 6],
            bytes[cursor + 7],
        ]) as usize;
        let value_kind = bytes[cursor + 8];

        let payload_end = cursor + 9 + key_len + value_len;
        let crc_end = payload_end + 4;
        if crc_end > bytes.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "sstable record body truncated",
            ));
        }

        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&bytes[cursor..payload_end]);
        let expected_crc = u32::from_le_bytes([
            bytes[payload_end],
            bytes[payload_end + 1],
            bytes[payload_end + 2],
            bytes[payload_end + 3],
        ]);
        if hasher.finalize() != expected_crc {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "sstable record CRC32C mismatch",
            ));
        }

        let key = bytes[cursor + 9..cursor + 9 + key_len].to_vec();
        let value_bytes = bytes[cursor + 9 + key_len..payload_end].to_vec();
        let value = match value_kind {
            VALUE_KIND_RAW => BlobValue::Raw(value_bytes),
            VALUE_KIND_COMPRESSED => BlobValue::Compressed(value_bytes),
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("sstable record has unknown value_kind {other:#04x}"),
                ));
            }
        };

        out.push((key, value));
        cursor = crc_end;
    }

    if u64::try_from(out.len()).unwrap_or(u64::MAX) != expected_count {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "sstable header claims {expected_count} records but body contains {}",
                out.len()
            ),
        ));
    }
    Ok(out)
}

/// Return the sibling ".tmp" path used during atomic rewrite.
fn tmp_sibling(path: &Path) -> PathBuf {
    let mut tmp = path.to_path_buf();
    let file_name = tmp
        .file_name()
        .map(std::ffi::OsStr::to_os_string)
        .unwrap_or_default();
    let mut with_suffix = file_name;
    with_suffix.push(".tmp");
    tmp.set_file_name(with_suffix);
    tmp
}

/// Read the current size of a footer'd `SSTable` file without loading its
/// records. Handy for the auto-flush decision that lives in
/// [`crate::blob::BlobStorage`].
///
/// Returns `Ok(None)` if the file does not exist.
///
/// # Errors
/// Propagates unexpected `io::Error`s.
pub fn sstable_size_on_disk(path: impl AsRef<Path>) -> io::Result<Option<u64>> {
    match std::fs::metadata(path) {
        Ok(m) => Ok(Some(m.len())),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

#[allow(dead_code)]
fn __sanity_check_footer_len() {
    // If FOOTER_LEN ever drifts away from `8 (records_size) + 8 (magic)`
    // the reader logic above will silently misread. This anchor exists
    // for the human next to touch the format constants; a proper
    // static assert lands with more format churn in α-3.3.
    let _: [u8; FOOTER_LEN] = [0_u8; 16];
}

// Silence an "unused" warning for `Seek` and `SeekFrom` when the tests
// module is compiled out. The imports would grow real usage in α-3.3
// when we add mmap-based random reads; for now they document the
// intent and shave a whitespace-only churn PR later.
#[allow(dead_code)]
fn __hold_seek_imports<T: Seek>(_: T, _: SeekFrom) {}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn sst_path(dir: &TempDir) -> PathBuf {
        dir.path().join("blob.sst")
    }

    #[test]
    fn roundtrip_empty_snapshot() {
        let tmp = TempDir::new().unwrap();
        let path = sst_path(&tmp);
        BlobSstable::write_from_iter::<'_, [(&[u8], &BlobValue); 0]>(&path, []).unwrap();

        let sst = BlobSstable::open(&path).unwrap().expect("file exists");
        assert!(sst.is_empty());
        assert_eq!(sst.len(), 0);
    }

    #[test]
    fn roundtrip_two_raw_records() {
        let tmp = TempDir::new().unwrap();
        let path = sst_path(&tmp);
        let raw_a = BlobValue::Raw(b"one".to_vec());
        let raw_b = BlobValue::Raw(b"two".to_vec());
        BlobSstable::write_from_iter(
            &path,
            [(b"a".as_slice(), &raw_a), (b"b".as_slice(), &raw_b)],
        )
        .unwrap();

        let sst = BlobSstable::open(&path).unwrap().unwrap();
        let recs: Vec<_> = sst.iter().collect();
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].0, b"a");
        assert_eq!(recs[0].1, &BlobValue::Raw(b"one".to_vec()));
        assert_eq!(recs[1].0, b"b");
        assert_eq!(recs[1].1, &BlobValue::Raw(b"two".to_vec()));
    }

    #[test]
    fn roundtrip_compressed_record_preserves_payload_verbatim() {
        let tmp = TempDir::new().unwrap();
        let path = sst_path(&tmp);
        // We do not care about the payload semantics here — the SSTable
        // must round-trip whatever bytes it was given.
        let compressed =
            BlobValue::Compressed(vec![0x78, 0x9c, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01]);
        BlobSstable::write_from_iter(&path, [(b"key".as_slice(), &compressed)]).unwrap();

        let sst = BlobSstable::open(&path).unwrap().unwrap();
        let recs: Vec<_> = sst.iter().collect();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].1, &compressed);
    }

    #[test]
    fn write_rejects_tombstone_entries() {
        let tmp = TempDir::new().unwrap();
        let path = sst_path(&tmp);
        let err = BlobSstable::write_from_iter(&path, [(b"k".as_slice(), &BlobValue::Tombstone)])
            .expect_err("tombstones are WAL-only");
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn open_returns_none_when_file_missing() {
        let tmp = TempDir::new().unwrap();
        assert!(BlobSstable::open(sst_path(&tmp)).unwrap().is_none());
    }

    #[test]
    fn corrupted_footer_magic_is_rejected() {
        let tmp = TempDir::new().unwrap();
        let path = sst_path(&tmp);
        let raw = BlobValue::Raw(b"v".to_vec());
        BlobSstable::write_from_iter(&path, [(b"k".as_slice(), &raw)]).unwrap();

        // Overwrite the last byte of the footer magic to break it.
        let mut bytes = std::fs::read(&path).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        std::fs::write(&path, &bytes).unwrap();

        let err = BlobSstable::open(&path).expect_err("corrupted footer");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn corrupted_record_crc_is_rejected() {
        let tmp = TempDir::new().unwrap();
        let path = sst_path(&tmp);
        let raw = BlobValue::Raw(b"value".to_vec());
        BlobSstable::write_from_iter(&path, [(b"key".as_slice(), &raw)]).unwrap();

        // Flip a bit inside the record payload (after the header, before the CRC).
        let mut bytes = std::fs::read(&path).unwrap();
        // Header is 24 bytes; first record starts there.
        bytes[HEADER_LEN + 9 + 1] ^= 0x01;
        std::fs::write(&path, &bytes).unwrap();

        let err = BlobSstable::open(&path).expect_err("corrupted CRC");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn size_on_disk_reports_none_for_missing_and_bytes_for_present() {
        let tmp = TempDir::new().unwrap();
        let path = sst_path(&tmp);
        assert!(sstable_size_on_disk(&path).unwrap().is_none());
        let raw = BlobValue::Raw(b"v".to_vec());
        BlobSstable::write_from_iter(&path, [(b"k".as_slice(), &raw)]).unwrap();
        let size = sstable_size_on_disk(&path).unwrap().unwrap();
        // Header (24) + 1 record (9 + 1 + 1 + 4) + footer (16) = 55 bytes.
        assert_eq!(size, 55);
    }
}
