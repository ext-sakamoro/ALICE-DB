//! Write-ahead log for the blob key-value store (v0.2.0-alpha.2).
//!
//! `BlobWal` is an append-only file backing the in-memory
//! [`crate::blob::BlobStorage`]. Every `put_blob` and `delete_blob`
//! invocation writes a checksum-guarded record here before mutating the
//! in-memory map, so a subsequent process can rebuild identical state
//! from the log alone.
//!
//! # Record layout
//!
//! Each record is framed as little-endian bytes. Multi-byte integers are
//! LE so the WAL is portable across host endianness.
//!
//! ```text
//! offset  size  field           description
//! ------  ----  --------------  ------------------------------------------------
//!  0      1     type            0x01 = Put, 0x02 = Delete
//!  1      4     key_len (u32)   payload key length in bytes
//!  5      4     value_len (u32) payload value length; 0 for Delete
//!  9      1     value_kind      0x00 = Raw, 0x01 = Compressed; ignored on Delete
//! 10      N     key             key_len bytes
//! 10+N    M     value           value_len bytes (absent on Delete)
//! trailer 4     crc32c          checksum over everything from offset 0 upward
//! ```
//!
//! # Truncation and recovery
//!
//! During [`BlobWal::replay`]:
//! - A short read at any point past a full record boundary is treated as
//!   a clean end-of-log (e.g. the process crashed mid-write). All fully
//!   framed records preceding the truncation are yielded.
//! - A record whose CRC32C does not match its payload is treated as
//!   corruption: the record is dropped and replay stops there (later
//!   records may exist but their offsets are no longer trustworthy).
//! - An empty or missing WAL file yields no records; the store opens
//!   with an empty state.
//!
//! # Concurrency and durability
//!
//! [`BlobWal::append_put`] / [`BlobWal::append_delete`] serialise the
//! record, write it in a single `write_all`, and then `sync_data` the
//! file handle before returning. Alpha-2 does one fsync per operation;
//! batched fsyncs land in alpha-3.
//!
//! No cross-process locking is applied. Alpha-2 assumes a single writer
//! per data directory. Multi-writer safety is deferred to alpha-3.

use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use parking_lot::Mutex;

use crate::blob::BlobValue;

/// Record type tag: `put` or `delete`.
const RECORD_TYPE_PUT: u8 = 0x01;
const RECORD_TYPE_DELETE: u8 = 0x02;

/// Value kind tag inside a Put record.
const VALUE_KIND_RAW: u8 = 0x00;
const VALUE_KIND_COMPRESSED: u8 = 0x01;

/// Fixed prefix before the variable-length key/value bytes:
///   1 (type) + 4 (`key_len`) + 4 (`value_len`) + 1 (`value_kind`)
const HEADER_LEN: usize = 10;

/// CRC32C trailer.
const CRC_LEN: usize = 4;

/// A single logical operation reconstructed from the WAL.
///
/// Emitted in file order by [`BlobWal::replay`]; callers apply them
/// left-to-right to arrive at the same in-memory state the original
/// writer would have produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WalRecord {
    /// Insert or overwrite `key` with the exact stored representation.
    /// The `BlobValue` variant is preserved verbatim (no re-encoding).
    Put(Vec<u8>, BlobValue),
    /// Tombstone a key. The corresponding in-memory entry becomes
    /// invisible to readers.
    Delete(Vec<u8>),
}

/// Append-only write-ahead log for the blob store.
///
/// Wraps a `File` behind a `parking_lot::Mutex` so multi-threaded writers
/// serialise their appends. The `Mutex` also guards the `sync_data`
/// call, so a return from `append_*` implies the record has been flushed
/// to the OS-level file system.
#[derive(Debug)]
pub struct BlobWal {
    path: PathBuf,
    file: Mutex<File>,
}

impl BlobWal {
    /// Open (or create) the WAL at `path`.
    ///
    /// Missing parent directories are created. The file is opened in
    /// read+write+append mode so replay reads and mutation writes share
    /// a single handle; every append seeks to end implicitly via the
    /// `append` flag.
    ///
    /// # Errors
    /// Returns `io::Error` if directory creation or file open fails.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .read(true)
            .create(true)
            .append(true)
            .open(&path)?;
        Ok(Self {
            path,
            file: Mutex::new(file),
        })
    }

    /// The on-disk location of the WAL. Handy for diagnostics.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Serialise a `put` record and durably append it.
    ///
    /// # Errors
    /// Any I/O failure from the underlying write or fsync propagates.
    pub fn append_put(&self, key: &[u8], value: &BlobValue) -> io::Result<()> {
        let (kind, bytes): (u8, &[u8]) = match value {
            BlobValue::Raw(b) => (VALUE_KIND_RAW, b.as_slice()),
            BlobValue::Compressed(b) => (VALUE_KIND_COMPRESSED, b.as_slice()),
            BlobValue::Tombstone => {
                // Callers must convert Tombstone -> Delete before
                // reaching the WAL; hitting this branch is a bug.
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "BlobWal::append_put received a Tombstone value; use append_delete",
                ));
            }
        };
        let frame = serialise_record(RECORD_TYPE_PUT, key, kind, bytes)?;
        self.write_frame(&frame)
    }

    /// Serialise a `delete` record and durably append it.
    ///
    /// # Errors
    /// Any I/O failure from the underlying write or fsync propagates.
    pub fn append_delete(&self, key: &[u8]) -> io::Result<()> {
        let frame = serialise_record(RECORD_TYPE_DELETE, key, VALUE_KIND_RAW, &[])?;
        self.write_frame(&frame)
    }

    fn write_frame(&self, frame: &[u8]) -> io::Result<()> {
        let mut file = self.file.lock();
        file.write_all(frame)?;
        // sync_data() is enough — we do not need metadata durability
        // (mtime, etc.) for correctness, only that the record bytes
        // survive a crash.
        file.sync_data()?;
        Ok(())
    }

    /// Read the entire WAL and yield every well-framed record in file order.
    ///
    /// Truncated tails are treated as clean end-of-log; corrupted (bad
    /// CRC) records stop replay early. See the module docs for the
    /// contract.
    ///
    /// # Errors
    /// Any read error (other than end-of-file, which is expected)
    /// propagates.
    pub fn replay(&self) -> io::Result<Vec<WalRecord>> {
        let mut file = self.file.lock();
        // Seek to start; the append handle keeps subsequent writes going
        // to the end regardless of where we leave the read cursor.
        file.seek(SeekFrom::Start(0))?;
        let mut reader = BufReader::new(&*file);

        let mut out = Vec::new();
        while let ReadOutcome::Record(rec) = read_one(&mut reader)? {
            out.push(rec);
        }
        Ok(out)
    }
}

/// Serialise one record: [header][key][value?][crc32c].
fn serialise_record(
    record_type: u8,
    key: &[u8],
    value_kind: u8,
    value: &[u8],
) -> io::Result<Vec<u8>> {
    let key_len: u32 = u32::try_from(key.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "key exceeds u32::MAX"))?;
    let value_len: u32 = u32::try_from(value.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "value exceeds u32::MAX"))?;

    let payload_len = HEADER_LEN + key.len() + value.len();
    let mut buf = Vec::with_capacity(payload_len + CRC_LEN);
    buf.push(record_type);
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

/// Outcome of trying to read one record from the WAL.
enum ReadOutcome {
    Record(WalRecord),
    Eof,
    Truncated,
    BadChecksum,
}

fn read_one<R: Read>(reader: &mut R) -> io::Result<ReadOutcome> {
    let mut header = [0_u8; HEADER_LEN];
    match read_exact_or_eof(reader, &mut header)? {
        ReadResult::Full => {}
        ReadResult::Eof => return Ok(ReadOutcome::Eof),
        ReadResult::Truncated => return Ok(ReadOutcome::Truncated),
    }

    let record_type = header[0];
    let key_len = u32::from_le_bytes([header[1], header[2], header[3], header[4]]) as usize;
    let value_len = u32::from_le_bytes([header[5], header[6], header[7], header[8]]) as usize;
    let value_kind = header[9];

    let mut key = vec![0_u8; key_len];
    match read_exact_or_eof(reader, &mut key)? {
        ReadResult::Full => {}
        ReadResult::Eof | ReadResult::Truncated => return Ok(ReadOutcome::Truncated),
    }

    let mut value = vec![0_u8; value_len];
    if value_len > 0 {
        match read_exact_or_eof(reader, &mut value)? {
            ReadResult::Full => {}
            ReadResult::Eof | ReadResult::Truncated => return Ok(ReadOutcome::Truncated),
        }
    }

    let mut crc_bytes = [0_u8; CRC_LEN];
    match read_exact_or_eof(reader, &mut crc_bytes)? {
        ReadResult::Full => {}
        ReadResult::Eof | ReadResult::Truncated => return Ok(ReadOutcome::Truncated),
    }
    let expected_crc = u32::from_le_bytes(crc_bytes);

    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&header);
    hasher.update(&key);
    hasher.update(&value);
    if hasher.finalize() != expected_crc {
        return Ok(ReadOutcome::BadChecksum);
    }

    match record_type {
        RECORD_TYPE_PUT => {
            let value = match value_kind {
                VALUE_KIND_RAW => BlobValue::Raw(value),
                VALUE_KIND_COMPRESSED => BlobValue::Compressed(value),
                _ => return Ok(ReadOutcome::BadChecksum),
            };
            Ok(ReadOutcome::Record(WalRecord::Put(key, value)))
        }
        RECORD_TYPE_DELETE => Ok(ReadOutcome::Record(WalRecord::Delete(key))),
        _ => Ok(ReadOutcome::BadChecksum),
    }
}

enum ReadResult {
    Full,
    Eof,
    Truncated,
}

fn read_exact_or_eof<R: Read>(reader: &mut R, buf: &mut [u8]) -> io::Result<ReadResult> {
    let mut read = 0;
    while read < buf.len() {
        match reader.read(&mut buf[read..]) {
            Ok(0) => {
                return Ok(if read == 0 {
                    ReadResult::Eof
                } else {
                    ReadResult::Truncated
                });
            }
            Ok(n) => read += n,
            Err(ref e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(ReadResult::Full)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn wal_path(dir: &TempDir) -> PathBuf {
        dir.path().join("blob.wal")
    }

    #[test]
    fn roundtrip_put_and_delete_records() {
        let tmp = TempDir::new().unwrap();
        let wal = BlobWal::open(wal_path(&tmp)).unwrap();
        wal.append_put(b"foo", &BlobValue::Raw(b"hello".to_vec()))
            .unwrap();
        wal.append_put(b"bar", &BlobValue::Compressed(vec![0x78, 0x9c, 0x03, 0x00]))
            .unwrap();
        wal.append_delete(b"foo").unwrap();

        let records = wal.replay().unwrap();
        assert_eq!(
            records,
            vec![
                WalRecord::Put(b"foo".to_vec(), BlobValue::Raw(b"hello".to_vec())),
                WalRecord::Put(
                    b"bar".to_vec(),
                    BlobValue::Compressed(vec![0x78, 0x9c, 0x03, 0x00]),
                ),
                WalRecord::Delete(b"foo".to_vec()),
            ]
        );
    }

    #[test]
    fn append_put_rejects_tombstone_variant() {
        let tmp = TempDir::new().unwrap();
        let wal = BlobWal::open(wal_path(&tmp)).unwrap();
        let err = wal
            .append_put(b"foo", &BlobValue::Tombstone)
            .expect_err("Tombstone must be routed through append_delete");
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn empty_wal_yields_no_records() {
        let tmp = TempDir::new().unwrap();
        let wal = BlobWal::open(wal_path(&tmp)).unwrap();
        let records = wal.replay().unwrap();
        assert!(records.is_empty());
    }

    #[test]
    fn truncated_tail_is_dropped_cleanly() {
        // Manually build two full records and half of a third, then
        // verify replay yields exactly the two intact ones.
        let tmp = TempDir::new().unwrap();
        let path = wal_path(&tmp);
        let wal = BlobWal::open(&path).unwrap();
        wal.append_put(b"k1", &BlobValue::Raw(b"v1".to_vec()))
            .unwrap();
        wal.append_put(b"k2", &BlobValue::Raw(b"v2".to_vec()))
            .unwrap();
        // Truncate the file at the size of ~1.5 records.
        let full_size = std::fs::metadata(&path).unwrap().len();
        // Each of our records is HEADER_LEN + 2 (key) + 2 (value) + CRC_LEN = 18 bytes.
        // Two full records = 36 bytes. Truncate to 27 to slice the second record.
        assert_eq!(full_size, 36);
        {
            let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
            f.set_len(27).unwrap();
        }
        let wal = BlobWal::open(&path).unwrap();
        let records = wal.replay().unwrap();
        assert_eq!(
            records,
            vec![WalRecord::Put(
                b"k1".to_vec(),
                BlobValue::Raw(b"v1".to_vec())
            )]
        );
    }

    #[test]
    fn corrupted_checksum_stops_replay_at_that_record() {
        let tmp = TempDir::new().unwrap();
        let path = wal_path(&tmp);
        let wal = BlobWal::open(&path).unwrap();
        wal.append_put(b"good", &BlobValue::Raw(b"ok".to_vec()))
            .unwrap();
        wal.append_put(b"bad", &BlobValue::Raw(b"no".to_vec()))
            .unwrap();
        // Corrupt one byte inside the second record's payload region.
        let bytes = std::fs::read(&path).unwrap();
        let first_record_len = HEADER_LEN + 4 + 2 + CRC_LEN; // 4-byte key "good" + 2-byte value "ok"
        let mut corrupted = bytes.clone();
        corrupted[first_record_len + HEADER_LEN + 1] ^= 0xff; // flip a bit inside "bad"
        std::fs::write(&path, corrupted).unwrap();

        let wal = BlobWal::open(&path).unwrap();
        let records = wal.replay().unwrap();
        assert_eq!(
            records,
            vec![WalRecord::Put(
                b"good".to_vec(),
                BlobValue::Raw(b"ok".to_vec())
            )]
        );
    }
}
