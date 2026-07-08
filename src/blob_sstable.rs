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

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use crate::blob::BlobValue;
use crate::bloom::BloomFilter;

/// File-format magic marking the start of a blob `SSTable`.
const HEADER_MAGIC: &[u8; 8] = b"ALICEBBS";
/// File-format magic marking the end of a blob `SSTable`.
const FOOTER_MAGIC: &[u8; 8] = b"ALICEEND";
/// v0.2.0-alpha.2 through v0.2.0-alpha.5: header + records + footer,
/// no Bloom section.
const FORMAT_VERSION_V1: u32 = 1;
/// v0.2.0-alpha.6+: header + records + Bloom section + extended footer
/// (`records_size` + `bloom_size` + magic).
const FORMAT_VERSION_V2: u32 = 2;
/// Every new write stamps this. Readers accept both v1 and v2.
const CURRENT_VERSION: u32 = FORMAT_VERSION_V2;
/// Bytes occupied by the fixed-size header.
const HEADER_LEN: usize = 24;
/// Bytes occupied by the v1 footer: `records_size (u64) + magic (8)`.
const FOOTER_LEN_V1: usize = 16;
/// Bytes occupied by the v2 footer: `records_size (u64) + bloom_size (u64) + magic (8)`.
const FOOTER_LEN_V2: usize = 24;
/// Encoded flag for `BlobValue::Raw`.
const VALUE_KIND_RAW: u8 = 0x00;
/// Encoded flag for `BlobValue::Compressed`.
const VALUE_KIND_COMPRESSED: u8 = 0x01;

/// Prelude of the Bloom section:
/// `num_bits (u64) + num_hashes (u32) + bloom_bytes_len (u64) + crc32c (u32)`.
const BLOOM_HEADER_LEN: usize = 8 + 4 + 8 + 4;

/// False-positive rate target for the Bloom filter attached to each
/// `SSTable`. 1% is the classic sweet spot: ~9.6 bits/key, 7 hashes.
const BLOOM_FALSE_POSITIVE_RATE: f64 = 0.01;

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
/// can be reconstructed via a single `extend`. Alpha-3.3b-1 additionally
/// carries the file's Bloom filter (when it is v2); a future alpha
/// (α-3.3b-2) will keep the file mmap'd for point lookups without full
/// load.
#[derive(Debug)]
pub struct BlobSstable {
    path: PathBuf,
    records: Vec<(Vec<u8>, BlobValue)>,
    /// Present only for format v2 files. `None` for v1 files written by
    /// v0.2.0-alpha.2 through v0.2.0-alpha.5 — callers must treat those
    /// as if the Bloom always accepts every key.
    bloom: Option<BloomFilter>,
}

impl BlobSstable {
    /// The on-disk location of the `SSTable`.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Bloom filter attached to this `SSTable`, if the file was written
    /// in format v2 or later. `None` when reading a legacy v1 file.
    ///
    /// Callers checking membership should treat `None` as
    /// "the Bloom cannot help — always look in the records".
    #[must_use]
    pub fn bloom(&self) -> Option<&BloomFilter> {
        self.bloom.as_ref()
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

        // Build the Bloom filter as we iterate — sized for this exact
        // record count and 1% false-positive rate.
        let mut bloom = BloomFilter::with_capacity(entries.len(), BLOOM_FALSE_POSITIVE_RATE);

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
            bloom.insert(key);
            let record = serialise_record(key, kind, bytes)?;
            writer.write_all(&record)?;
            records_size = records_size
                .checked_add(u64::try_from(record.len()).unwrap_or(u64::MAX))
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "records section overflows u64")
                })?;
        }

        // Bloom section (v2). We record its total on-disk length so the
        // reader can partition the file without walking every record.
        let bloom_section = serialise_bloom(&bloom)?;
        let bloom_size = u64::try_from(bloom_section.len()).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "bloom section overflows u64")
        })?;
        writer.write_all(&bloom_section)?;

        // v2 Footer: records_size + bloom_size + magic.
        writer.write_all(&records_size.to_le_bytes())?;
        writer.write_all(&bloom_size.to_le_bytes())?;
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

        // Read the whole file — alpha-3.2a/α-3.3b-1 are not memory-frugal
        // by design. α-3.3b-2 will keep the file mmap'd instead.
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;

        if bytes.len() < HEADER_LEN + FOOTER_LEN_V1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "sstable is smaller than the fixed header + v1 footer",
            ));
        }
        if &bytes[0..8] != HEADER_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "sstable header magic mismatch",
            ));
        }
        let version = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let num_records = u64::from_le_bytes([
            bytes[12], bytes[13], bytes[14], bytes[15], bytes[16], bytes[17], bytes[18], bytes[19],
        ]);

        let (records_slice, bloom) = match version {
            FORMAT_VERSION_V1 => {
                let footer_start = bytes.len() - FOOTER_LEN_V1;
                if &bytes[footer_start + 8..] != FOOTER_MAGIC {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "sstable v1 footer magic mismatch",
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
                        "sstable v1 footer records_size does not match section length",
                    ));
                }
                (records_slice, None)
            }
            FORMAT_VERSION_V2 => {
                if bytes.len() < HEADER_LEN + FOOTER_LEN_V2 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "sstable v2 is smaller than the fixed header + v2 footer",
                    ));
                }
                let footer_start = bytes.len() - FOOTER_LEN_V2;
                if &bytes[footer_start + 16..] != FOOTER_MAGIC {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "sstable v2 footer magic mismatch",
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
                let footer_bloom_size = u64::from_le_bytes([
                    bytes[footer_start + 8],
                    bytes[footer_start + 9],
                    bytes[footer_start + 10],
                    bytes[footer_start + 11],
                    bytes[footer_start + 12],
                    bytes[footer_start + 13],
                    bytes[footer_start + 14],
                    bytes[footer_start + 15],
                ]);
                let records_end = HEADER_LEN
                    + usize::try_from(footer_records_size).map_err(|_| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "sstable v2 records_size exceeds usize",
                        )
                    })?;
                let bloom_end = records_end
                    + usize::try_from(footer_bloom_size).map_err(|_| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "sstable v2 bloom_size exceeds usize",
                        )
                    })?;
                if bloom_end != footer_start {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "sstable v2 section sizes do not add up to file size",
                    ));
                }
                let records_slice = &bytes[HEADER_LEN..records_end];
                let bloom_slice = &bytes[records_end..bloom_end];
                let bloom = parse_bloom(bloom_slice)?;
                (records_slice, Some(bloom))
            }
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unsupported sstable version {other}; expected 1 or 2"),
                ));
            }
        };

        let records = parse_records(records_slice, num_records)?;
        Ok(Some(Self {
            path,
            records,
            bloom,
        }))
    }
}

/// Byte offset + shape of a single record within a mmap'd `SSTable`.
///
/// Introduced in v0.2.0-alpha.7 alongside [`LoadedSstable`]. The offset
/// index lives in memory (one entry per record) but the *value bytes*
/// stay on disk, faulted in only when a read actually touches them.
#[derive(Debug, Clone, Copy)]
struct RecordOffset {
    /// Byte offset of the record header (`key_len`, `value_len`,
    /// `value_kind`) within the mmap.
    record_start: u64,
    key_len: u32,
    value_len: u32,
    value_kind: u8,
}

/// An `SSTable` loaded via memory-mapped I/O.
///
/// Introduced in v0.2.0-alpha.7 as the read-path replacement for
/// [`BlobSstable`]'s materialised `records` vector. The mmap keeps the
/// file's bytes accessible without pulling them all into the process
/// heap; only an in-memory key → offset index is retained.
///
/// Point lookups consult the Bloom filter first (when present), so keys
/// that were never inserted are rejected without touching the index —
/// let alone the mmap. When the key survives the Bloom, the index
/// resolves it to an offset in one `BTreeMap` lookup and the value
/// bytes are copied out of the mmap on demand.
///
/// # Legacy v1 files
///
/// Files written by v0.2.0-alpha.2 through v0.2.0-alpha.5 have no Bloom
/// section; [`LoadedSstable::bloom`] reports `None` and lookups fall
/// through to the index directly (correct, just not accelerated).
pub struct LoadedSstable {
    path: PathBuf,
    /// Kept alive for the lifetime of the mmap. Some platforms release
    /// the file handle when it drops; the mmap must outlive it.
    _file: File,
    mmap: Mmap,
    bloom: Option<BloomFilter>,
    /// Key → offset within the mmap. Built once at open time.
    index: BTreeMap<Vec<u8>, RecordOffset>,
}

impl std::fmt::Debug for LoadedSstable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `mmap`, `_file`, and `index` are intentionally elided — the
        // raw bytes and the offset table are noisy in log output. We
        // still surface the summary that matters (file size, record
        // count, whether the file carries a Bloom).
        f.debug_struct("LoadedSstable")
            .field("path", &self.path)
            .field("mmap_len", &self.mmap.len())
            .field("bloom", &self.bloom.is_some())
            .field("records", &self.index.len())
            .finish_non_exhaustive()
    }
}

impl LoadedSstable {
    /// Open the `SSTable` at `path` via mmap. Returns `Ok(None)` if the
    /// file does not exist (mirrors [`BlobSstable::open`] semantics).
    ///
    /// The whole records section is walked once to verify per-record
    /// CRCs and build the offset index. After that point-lookups touch
    /// only the mmap + the `BTreeMap`.
    ///
    /// # Errors
    /// Returns an `io::Error` if the file exists but is malformed
    /// (header/footer magic, section sizes, per-record CRC, or Bloom
    /// section CRC). The file is treated as corrupt in that case.
    ///
    /// # Safety
    /// The mmap is created via `memmap2::Mmap::map`, which relies on the
    /// underlying file *not* being modified while mapped. Our writer
    /// path uses `.tmp` + `rename`, so the file we mmap is never
    /// mutated in place. Concurrent readers within the same process
    /// observe consistent bytes; the file may safely be unlinked while
    /// still mapped on Unix.
    pub fn open_mmap(path: impl AsRef<Path>) -> io::Result<Option<Self>> {
        let path = path.as_ref().to_path_buf();
        let file = match File::open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e),
        };

        // SAFETY: see the `Safety` note above.
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < HEADER_LEN + FOOTER_LEN_V1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "sstable is smaller than the fixed header + v1 footer",
            ));
        }
        if &mmap[0..8] != HEADER_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "sstable header magic mismatch",
            ));
        }
        let version = u32::from_le_bytes([mmap[8], mmap[9], mmap[10], mmap[11]]);
        let num_records = u64::from_le_bytes([
            mmap[12], mmap[13], mmap[14], mmap[15], mmap[16], mmap[17], mmap[18], mmap[19],
        ]);

        let (records_end, bloom) = match version {
            FORMAT_VERSION_V1 => {
                let footer_start = mmap.len() - FOOTER_LEN_V1;
                if &mmap[footer_start + 8..] != FOOTER_MAGIC {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "sstable v1 footer magic mismatch",
                    ));
                }
                let footer_records_size = u64::from_le_bytes([
                    mmap[footer_start],
                    mmap[footer_start + 1],
                    mmap[footer_start + 2],
                    mmap[footer_start + 3],
                    mmap[footer_start + 4],
                    mmap[footer_start + 5],
                    mmap[footer_start + 6],
                    mmap[footer_start + 7],
                ]);
                let end = HEADER_LEN
                    + usize::try_from(footer_records_size).map_err(|_| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "sstable v1 records_size exceeds usize",
                        )
                    })?;
                if end != footer_start {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "sstable v1 records_size does not match section length",
                    ));
                }
                (end, None)
            }
            FORMAT_VERSION_V2 => {
                if mmap.len() < HEADER_LEN + FOOTER_LEN_V2 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "sstable v2 is smaller than the fixed header + v2 footer",
                    ));
                }
                let footer_start = mmap.len() - FOOTER_LEN_V2;
                if &mmap[footer_start + 16..] != FOOTER_MAGIC {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "sstable v2 footer magic mismatch",
                    ));
                }
                let footer_records_size = u64::from_le_bytes([
                    mmap[footer_start],
                    mmap[footer_start + 1],
                    mmap[footer_start + 2],
                    mmap[footer_start + 3],
                    mmap[footer_start + 4],
                    mmap[footer_start + 5],
                    mmap[footer_start + 6],
                    mmap[footer_start + 7],
                ]);
                let footer_bloom_size = u64::from_le_bytes([
                    mmap[footer_start + 8],
                    mmap[footer_start + 9],
                    mmap[footer_start + 10],
                    mmap[footer_start + 11],
                    mmap[footer_start + 12],
                    mmap[footer_start + 13],
                    mmap[footer_start + 14],
                    mmap[footer_start + 15],
                ]);
                let records_end = HEADER_LEN
                    + usize::try_from(footer_records_size).map_err(|_| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "sstable v2 records_size exceeds usize",
                        )
                    })?;
                let bloom_end = records_end
                    + usize::try_from(footer_bloom_size).map_err(|_| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "sstable v2 bloom_size exceeds usize",
                        )
                    })?;
                if bloom_end != footer_start {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "sstable v2 section sizes do not add up to file size",
                    ));
                }
                let bloom = parse_bloom(&mmap[records_end..bloom_end])?;
                (records_end, Some(bloom))
            }
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unsupported sstable version {other}; expected 1 or 2"),
                ));
            }
        };

        let index = build_offset_index(&mmap[HEADER_LEN..records_end], num_records, HEADER_LEN)?;

        Ok(Some(Self {
            path,
            _file: file,
            mmap,
            bloom,
            index,
        }))
    }

    /// On-disk location of the file backing this `SSTable`.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The Bloom filter attached to this file, if any (format v2+).
    #[must_use]
    pub fn bloom(&self) -> Option<&BloomFilter> {
        self.bloom.as_ref()
    }

    /// Number of records in the file.
    #[must_use]
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Whether the file contains any records.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Point lookup by key.
    ///
    /// Consults the Bloom filter first (when present); on a Bloom miss
    /// returns `None` without touching the index. On a Bloom hit
    /// resolves the offset in one `BTreeMap` lookup and copies the
    /// value bytes out of the mmap.
    ///
    /// Returns `None` for absent keys. `SSTable` files never store
    /// [`BlobValue::Tombstone`], so `Some(...)` always carries a live
    /// value.
    #[must_use]
    pub fn get(&self, key: &[u8]) -> Option<BlobValue> {
        if let Some(bloom) = &self.bloom {
            if !bloom.contains(key) {
                return None;
            }
        }
        let offset = self.index.get(key)?;
        self.value_at(offset)
    }

    /// Iterate every `(key, value)` pair in ascending key order.
    ///
    /// Value bytes are copied out of the mmap into an owned `BlobValue`
    /// per element (unavoidable while `BlobValue` owns its bytes; a
    /// future iteration could expose a borrowing view).
    pub fn iter(&self) -> impl Iterator<Item = (&[u8], BlobValue)> + '_ {
        self.index
            .iter()
            .filter_map(|(k, offset)| self.value_at(offset).map(|v| (k.as_slice(), v)))
    }

    /// Iterate `(key, value)` pairs whose keys start with `prefix`, in
    /// ascending key order.
    pub fn iter_prefix<'a>(
        &'a self,
        prefix: &'a [u8],
    ) -> impl Iterator<Item = (&'a [u8], BlobValue)> + 'a {
        self.index
            .range(prefix.to_vec()..)
            .take_while(move |(k, _)| k.starts_with(prefix))
            .filter_map(move |(k, offset)| self.value_at(offset).map(|v| (k.as_slice(), v)))
    }

    /// Materialise the [`BlobValue`] at `offset` by copying bytes out
    /// of the mmap. Returns `None` only if the stored `value_kind` byte
    /// is unrecognised (which would indicate on-disk corruption not
    /// caught at open time — should not happen in practice).
    fn value_at(&self, offset: &RecordOffset) -> Option<BlobValue> {
        let record_start = usize::try_from(offset.record_start).ok()?;
        let key_len = offset.key_len as usize;
        let value_len = offset.value_len as usize;
        let value_start = record_start.checked_add(9)?.checked_add(key_len)?;
        let value_end = value_start.checked_add(value_len)?;
        if value_end > self.mmap.len() {
            return None;
        }
        let bytes = self.mmap[value_start..value_end].to_vec();
        match offset.value_kind {
            VALUE_KIND_RAW => Some(BlobValue::Raw(bytes)),
            VALUE_KIND_COMPRESSED => Some(BlobValue::Compressed(bytes)),
            _ => None,
        }
    }
}

/// Walk the records section once, verifying each per-record CRC and
/// recording each key's byte offset within the mmap.
///
/// `records_slice` is the raw records section (between the fixed header
/// and the footer / Bloom section). `records_slice_file_offset` is the
/// offset of `records_slice[0]` within the mmap so the returned
/// [`RecordOffset::record_start`] values are file-relative.
fn build_offset_index(
    records_slice: &[u8],
    expected_count: u64,
    records_slice_file_offset: usize,
) -> io::Result<BTreeMap<Vec<u8>, RecordOffset>> {
    let mut out: BTreeMap<Vec<u8>, RecordOffset> = BTreeMap::new();
    let mut cursor = 0_usize;
    let mut seen: u64 = 0;
    while cursor < records_slice.len() {
        if records_slice.len() - cursor < 9 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "sstable record header truncated",
            ));
        }
        let key_len = u32::from_le_bytes([
            records_slice[cursor],
            records_slice[cursor + 1],
            records_slice[cursor + 2],
            records_slice[cursor + 3],
        ]);
        let value_len = u32::from_le_bytes([
            records_slice[cursor + 4],
            records_slice[cursor + 5],
            records_slice[cursor + 6],
            records_slice[cursor + 7],
        ]);
        let value_kind = records_slice[cursor + 8];

        let key_end = cursor + 9 + key_len as usize;
        let payload_end = key_end + value_len as usize;
        let crc_end = payload_end + 4;
        if crc_end > records_slice.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "sstable record body truncated",
            ));
        }

        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&records_slice[cursor..payload_end]);
        let expected_crc = u32::from_le_bytes([
            records_slice[payload_end],
            records_slice[payload_end + 1],
            records_slice[payload_end + 2],
            records_slice[payload_end + 3],
        ]);
        if hasher.finalize() != expected_crc {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "sstable record CRC32C mismatch",
            ));
        }
        if value_kind != VALUE_KIND_RAW && value_kind != VALUE_KIND_COMPRESSED {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("sstable record has unknown value_kind {value_kind:#04x}"),
            ));
        }

        let key = records_slice[cursor + 9..key_end].to_vec();
        let record_start_file = records_slice_file_offset + cursor;
        let record_start = u64::try_from(record_start_file).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "sstable record offset exceeds u64",
            )
        })?;
        out.insert(
            key,
            RecordOffset {
                record_start,
                key_len,
                value_len,
                value_kind,
            },
        );
        seen += 1;
        cursor = crc_end;
    }
    if seen != expected_count {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("sstable header claims {expected_count} records but body contains {seen}"),
        ));
    }
    Ok(out)
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

/// Encode the Bloom filter as a byte sequence with a CRC32C trailer.
/// Layout: `num_bits (u64) + num_hashes (u32) + bloom_bytes_len (u64) + bloom_bits + crc32c`.
fn serialise_bloom(bloom: &BloomFilter) -> io::Result<Vec<u8>> {
    let bits = bloom.as_bits();
    let bits_len = u64::try_from(bits.len()).map_err(|_| {
        io::Error::new(io::ErrorKind::InvalidInput, "bloom bits length exceeds u64")
    })?;
    let mut buf = Vec::with_capacity(BLOOM_HEADER_LEN + bits.len());
    buf.extend_from_slice(&bloom.num_bits().to_le_bytes());
    buf.extend_from_slice(&bloom.num_hashes().to_le_bytes());
    buf.extend_from_slice(&bits_len.to_le_bytes());
    buf.extend_from_slice(bits);
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&buf);
    let crc = hasher.finalize();
    buf.extend_from_slice(&crc.to_le_bytes());
    Ok(buf)
}

/// Decode a Bloom section produced by [`serialise_bloom`].
fn parse_bloom(bytes: &[u8]) -> io::Result<BloomFilter> {
    if bytes.len() < BLOOM_HEADER_LEN {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "sstable bloom section shorter than its header",
        ));
    }
    let num_bits = u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]);
    let num_hashes = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    let bits_len = u64::from_le_bytes([
        bytes[12], bytes[13], bytes[14], bytes[15], bytes[16], bytes[17], bytes[18], bytes[19],
    ]);
    let bits_len_usize = usize::try_from(bits_len).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "sstable bloom bits length exceeds usize",
        )
    })?;
    let expected_total = BLOOM_HEADER_LEN + bits_len_usize;
    if bytes.len() != expected_total {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "sstable bloom section length mismatch",
        ));
    }
    // Verify CRC over everything up to (but not including) the trailer.
    let crc_start = expected_total - 4;
    let expected_crc = u32::from_le_bytes([
        bytes[crc_start],
        bytes[crc_start + 1],
        bytes[crc_start + 2],
        bytes[crc_start + 3],
    ]);
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&bytes[..crc_start]);
    if hasher.finalize() != expected_crc {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "sstable bloom CRC32C mismatch",
        ));
    }
    // Bits section: after the fixed header, before the CRC.
    let bits_start = 8 + 4 + 8;
    let bits_end = bits_start + bits_len_usize;
    let bits = bytes[bits_start..bits_end].to_vec();
    Ok(BloomFilter::from_raw(bits, num_bits, num_hashes))
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
    // If either footer length ever drifts away from its documented
    // layout the reader logic above will silently misread. These
    // anchors exist for the human next to touch the format constants.
    let _: [u8; FOOTER_LEN_V1] = [0_u8; 16];
    let _: [u8; FOOTER_LEN_V2] = [0_u8; 24];
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
        // Format v2 (α-3.3b-1):
        //   Header (24)
        //   Records = 1 × (9 + key_len=1 + value_len=1 + crc=4) = 15
        //   Bloom section = 8 + 4 + 8 + bits_len (~2 for n=1) + crc=4 = ~26
        //   Footer (24 = records_size + bloom_size + magic)
        // = 89 bytes. Anchor is loose to survive the Bloom sizing
        // formula tweaking that may land in α-3.3b-2.
        assert!(
            (80..=100).contains(&size),
            "expected a v2 file in the 80..=100 byte range; got {size}"
        );
    }

    // ---------- LoadedSstable (v0.2.0-alpha.7) ----------

    /// Write a small v2 file with a single Raw record, mmap-open it, and
    /// probe a couple of round-trip properties.
    #[test]
    fn loaded_sstable_opens_and_serves_raw_value_via_mmap() {
        let tmp = TempDir::new().unwrap();
        let path = sst_path(&tmp);
        let raw = BlobValue::Raw(b"hello".to_vec());
        BlobSstable::write_from_iter(&path, [(b"greeting".as_slice(), &raw)]).unwrap();

        let sst = LoadedSstable::open_mmap(&path).unwrap().unwrap();
        assert_eq!(sst.len(), 1);
        assert!(!sst.is_empty());
        assert_eq!(sst.path(), path.as_path());
        assert!(sst.bloom().is_some(), "v2 files must expose a Bloom");

        match sst.get(b"greeting").expect("key must resolve") {
            BlobValue::Raw(bytes) => assert_eq!(bytes, b"hello"),
            other => panic!("expected Raw, got {other:?}"),
        }
    }

    #[test]
    fn loaded_sstable_serves_compressed_value_bytes_verbatim() {
        let tmp = TempDir::new().unwrap();
        let path = sst_path(&tmp);
        // Any bytes work; the SSTable does not decompress them.
        let compressed_bytes = vec![0x78, 0x9c, 0xab, 0xcd, 0xef, 0x00, 0x01, 0x02];
        let comp = BlobValue::Compressed(compressed_bytes.clone());
        BlobSstable::write_from_iter(&path, [(b"payload".as_slice(), &comp)]).unwrap();

        let sst = LoadedSstable::open_mmap(&path).unwrap().unwrap();
        match sst.get(b"payload").unwrap() {
            BlobValue::Compressed(bytes) => assert_eq!(bytes, compressed_bytes),
            other => panic!("expected Compressed, got {other:?}"),
        }
    }

    #[test]
    fn loaded_sstable_reports_none_for_absent_key() {
        let tmp = TempDir::new().unwrap();
        let path = sst_path(&tmp);
        let raw = BlobValue::Raw(b"v".to_vec());
        BlobSstable::write_from_iter(&path, [(b"present".as_slice(), &raw)]).unwrap();
        let sst = LoadedSstable::open_mmap(&path).unwrap().unwrap();
        assert!(
            sst.get(b"absent").is_none(),
            "absent key must resolve to None"
        );
    }

    #[test]
    fn loaded_sstable_iter_returns_all_records_in_sorted_order() {
        let tmp = TempDir::new().unwrap();
        let path = sst_path(&tmp);
        let a = BlobValue::Raw(b"1".to_vec());
        let b = BlobValue::Raw(b"2".to_vec());
        let c = BlobValue::Raw(b"3".to_vec());
        // Keys must be sorted for write_from_iter to accept them; here
        // they already are.
        BlobSstable::write_from_iter(
            &path,
            [
                (b"alpha".as_slice(), &a),
                (b"beta".as_slice(), &b),
                (b"gamma".as_slice(), &c),
            ],
        )
        .unwrap();
        let sst = LoadedSstable::open_mmap(&path).unwrap().unwrap();

        let observed: Vec<(Vec<u8>, BlobValue)> =
            sst.iter().map(|(k, v)| (k.to_vec(), v)).collect();
        assert_eq!(observed.len(), 3);
        assert_eq!(observed[0].0, b"alpha");
        assert_eq!(observed[1].0, b"beta");
        assert_eq!(observed[2].0, b"gamma");
    }

    #[test]
    fn loaded_sstable_iter_prefix_walks_matching_keys_only() {
        let tmp = TempDir::new().unwrap();
        let path = sst_path(&tmp);
        let v = BlobValue::Raw(b"x".to_vec());
        BlobSstable::write_from_iter(
            &path,
            [
                (b"cat".as_slice(), &v),
                (b"catamaran".as_slice(), &v),
                (b"category".as_slice(), &v),
                (b"dog".as_slice(), &v),
            ],
        )
        .unwrap();
        let sst = LoadedSstable::open_mmap(&path).unwrap().unwrap();

        let observed: Vec<Vec<u8>> = sst.iter_prefix(b"cat").map(|(k, _)| k.to_vec()).collect();
        assert_eq!(
            observed,
            vec![b"cat".to_vec(), b"catamaran".to_vec(), b"category".to_vec()],
        );
    }

    /// Format v1 file (hand-written to alpha.5 bytes) must open through
    /// the mmap path with `bloom() == None` and still surface records.
    #[test]
    fn loaded_sstable_opens_legacy_v1_files_without_bloom() {
        let tmp = TempDir::new().unwrap();
        let path = sst_path(&tmp);

        // Manually assemble a v1 file with one Raw record.
        let key = b"legacy";
        let value = b"value";
        let mut record: Vec<u8> = Vec::new();
        record.extend_from_slice(&u32::to_le_bytes(u32::try_from(key.len()).unwrap()));
        record.extend_from_slice(&u32::to_le_bytes(u32::try_from(value.len()).unwrap()));
        record.push(0x00); // Raw
        record.extend_from_slice(key);
        record.extend_from_slice(value);
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&record);
        let crc = hasher.finalize();
        record.extend_from_slice(&crc.to_le_bytes());

        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(HEADER_MAGIC).unwrap();
        file.write_all(&FORMAT_VERSION_V1.to_le_bytes()).unwrap();
        file.write_all(&1_u64.to_le_bytes()).unwrap();
        file.write_all(&0_u32.to_le_bytes()).unwrap();
        file.write_all(&record).unwrap();
        let records_size = u64::try_from(record.len()).unwrap();
        file.write_all(&records_size.to_le_bytes()).unwrap();
        file.write_all(FOOTER_MAGIC).unwrap();
        file.sync_all().unwrap();
        drop(file);

        let sst = LoadedSstable::open_mmap(&path).unwrap().unwrap();
        assert!(sst.bloom().is_none(), "v1 files predate the Bloom section");
        assert_eq!(sst.len(), 1);
        match sst.get(b"legacy").unwrap() {
            BlobValue::Raw(bytes) => assert_eq!(bytes, b"value"),
            other => panic!("expected Raw, got {other:?}"),
        }
    }
}
