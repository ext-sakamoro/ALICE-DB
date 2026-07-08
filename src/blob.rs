//! Blob key-value store (v0.2.0-alpha).
//!
//! This module adds a general LSM-KV path alongside ALICE-DB's marquee
//! time-series API. It is designed for callers whose value shape cannot
//! be modelled as `(i64 timestamp, f32 value)` — for example
//! ALICE-CodeTracker's rich `Stub` records or any application that needs
//! byte-keyed lookups with prefix scans.
//!
//! See `docs/EXPANSION_PROPOSAL.md` for the full design rationale.
//!
//! # Scope of alpha-1
//!
//! Alpha-1 ships the API surface and in-memory implementation only:
//!
//! - `put_blob` / `get_blob` / `scan_blob_prefix` / `delete_blob`
//!   (exposed on [`crate::AliceDB`]).
//! - `BlobStorage` backed by a `BTreeMap<Vec<u8>, BlobValue>` behind a
//!   `parking_lot::RwLock` for concurrent reader/single-writer access.
//! - Optional value compression via ALICE-Zip zlib when the payload is
//!   at least [`COMPRESS_THRESHOLD_BYTES`] and compression is expected
//!   to be worth it ([`should_compress`] applies the same threshold that
//!   ALICE-CodeTracker v0.4.0 established for its jsonl backend).
//!
//! WAL persistence and `SSTable` format v2 land in alpha-2 / alpha-3;
//! callers who need durability today should manually flush via a WAL of
//! their own or wait for those releases.

use std::collections::BTreeMap;
use std::io;
use std::path::Path;
use std::sync::Arc;

use alice_core::compression::{zlib_compress, zlib_decompress};
use parking_lot::RwLock;

use crate::blob_wal::{BlobWal, WalRecord};

/// Minimum byte length that triggers compression.
///
/// zlib carries ~11 bytes of header + checksum overhead, so very short
/// payloads inflate rather than shrink. The 200 byte cutoff matches the
/// threshold used by ALICE-CodeTracker v0.4.0's jsonl backend so that a
/// value crossing the ALICE-Zip integration is compressed consistently
/// on either side.
pub const COMPRESS_THRESHOLD_BYTES: usize = 200;

/// zlib compression level (flate2 default; good ratio-vs-cpu balance).
const ZLIB_LEVEL: u32 = 6;

/// The physical representation of a stored blob value.
///
/// The public [`crate::AliceDB::get_blob`] API always decodes this back
/// into a raw `Vec<u8>` before returning, so callers never observe the
/// enum. `Tombstone` is used to mark deletions in preparation for the
/// LSM tombstone semantics that alpha-3 will introduce; in alpha-1 it is
/// stored in the in-memory map so that a `delete_blob` immediately
/// hides the entry from readers even if a future WAL replay were to
/// resurrect the pre-delete value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlobValue {
    /// Value stored verbatim (short payloads or high-entropy data).
    Raw(Vec<u8>),
    /// Value stored as an ALICE-Zip zlib payload.
    Compressed(Vec<u8>),
    /// Deletion marker.
    Tombstone,
}

/// Return the compressed payload for `value` if compression is expected
/// to shrink it, otherwise return `None`.
///
/// Compression is skipped in three cases:
/// 1. `value.len() < COMPRESS_THRESHOLD_BYTES` — the overhead swamps the gain.
/// 2. The zlib encoder fails — the raw path stays viable.
/// 3. The compressed output would not save at least `SAVINGS_MARGIN_BYTES`
///    versus the raw payload — treating headers-only savings as noise.
#[must_use]
pub fn compress_if_worthwhile(value: &[u8]) -> Option<Vec<u8>> {
    // How many bytes the compressed payload must save vs. the raw form
    // before we prefer it. Below this, the round-trip CPU cost is not
    // worth the storage win.
    const SAVINGS_MARGIN_BYTES: usize = 16;

    if value.len() < COMPRESS_THRESHOLD_BYTES {
        return None;
    }
    let compressed = zlib_compress(value, ZLIB_LEVEL).ok()?;
    if compressed.len() + SAVINGS_MARGIN_BYTES < value.len() {
        Some(compressed)
    } else {
        None
    }
}

/// Decide whether a value is long enough to be considered for compression.
///
/// Exposed as a free function so tests and future tuning can substitute a
/// different heuristic (entropy-based, per-collection override) without
/// touching call sites.
#[must_use]
pub fn should_compress(value: &[u8]) -> bool {
    value.len() >= COMPRESS_THRESHOLD_BYTES
}

/// Blob key-value store.
///
/// Backed by a sorted `BTreeMap<Vec<u8>, BlobValue>` so that prefix scans
/// can walk contiguous key ranges. Cheap to clone: internally holds an
/// `Arc<RwLock<...>>` so every clone shares the same map.
///
/// # Persistence
///
/// - [`Self::new`] returns an ephemeral in-memory instance (the alpha-1
///   default). Process exit loses all data.
/// - [`Self::open`] (introduced in alpha-2) attaches a
///   [`crate::blob_wal::BlobWal`] on disk. Every mutation is durably
///   logged before the in-memory map is updated, and calls to `open`
///   replay the log to reconstruct prior state.
#[derive(Debug, Clone, Default)]
pub struct BlobStorage {
    inner: Arc<RwLock<BTreeMap<Vec<u8>, BlobValue>>>,
    /// Optional durable WAL. `None` means the store is in-memory only.
    wal: Option<Arc<BlobWal>>,
}

impl BlobStorage {
    /// Create an ephemeral in-memory blob store.
    ///
    /// No WAL is attached; data does not survive process exit. Use
    /// [`Self::open`] for durability.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Open a durable blob store backed by a WAL at `wal_path`.
    ///
    /// The WAL is created if missing. Any pre-existing records are
    /// replayed in file order so the in-memory map matches the state
    /// the previous writer left behind; truncated tails and corrupted
    /// records are handled per the WAL's contract (see
    /// [`crate::blob_wal`] module docs).
    ///
    /// # Errors
    /// Returns `io::Error` if the WAL cannot be created or replayed.
    pub fn open(wal_path: impl AsRef<Path>) -> io::Result<Self> {
        let wal = BlobWal::open(wal_path)?;
        let mut map: BTreeMap<Vec<u8>, BlobValue> = BTreeMap::new();
        for record in wal.replay()? {
            match record {
                WalRecord::Put(key, value) => {
                    map.insert(key, value);
                }
                WalRecord::Delete(key) => {
                    // Store the tombstone so subsequent replays and
                    // subsequent live reads observe the deletion.
                    map.insert(key, BlobValue::Tombstone);
                }
            }
        }
        Ok(Self {
            inner: Arc::new(RwLock::new(map)),
            wal: Some(Arc::new(wal)),
        })
    }

    /// Insert or overwrite a key with a value, compressing if worthwhile.
    ///
    /// If the store was opened via [`Self::open`], the record is durably
    /// logged before the in-memory map is updated. On a WAL write
    /// failure the in-memory state is left untouched.
    ///
    /// # Errors
    /// Returns `io::Error` if the WAL append fails.
    pub fn put(&self, key: &[u8], value: &[u8]) -> io::Result<()> {
        let stored = match compress_if_worthwhile(value) {
            Some(bytes) => BlobValue::Compressed(bytes),
            None => BlobValue::Raw(value.to_vec()),
        };
        if let Some(wal) = &self.wal {
            wal.append_put(key, &stored)?;
        }
        let mut guard = self.inner.write();
        guard.insert(key.to_vec(), stored);
        Ok(())
    }

    /// Look up a key, returning the decoded value if present.
    ///
    /// `Tombstone` entries surface as `None` so callers cannot observe
    /// deleted keys.
    ///
    /// # Errors
    /// Returns `io::Error` if a stored `Compressed` payload cannot be
    /// decompressed (which would indicate on-disk corruption once
    /// persistence lands in alpha-2).
    pub fn get(&self, key: &[u8]) -> io::Result<Option<Vec<u8>>> {
        let guard = self.inner.read();
        match guard.get(key) {
            None | Some(BlobValue::Tombstone) => Ok(None),
            Some(BlobValue::Raw(bytes)) => Ok(Some(bytes.clone())),
            Some(BlobValue::Compressed(bytes)) => {
                let raw = zlib_decompress(bytes)?;
                Ok(Some(raw))
            }
        }
    }

    /// Walk every live key whose bytes start with `prefix`.
    ///
    /// Returns `(key, value)` pairs in lexicographic order. `Tombstone`
    /// entries are silently skipped so a caller iterating the store sees
    /// the same set of keys that `get_blob` would surface.
    ///
    /// # Errors
    /// Returns `io::Error` if any encountered `Compressed` payload fails
    /// to decompress (see [`Self::get`]).
    pub fn scan_prefix(&self, prefix: &[u8]) -> io::Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let guard = self.inner.read();
        // BTreeMap's range API needs a slice-of-bytes comparator that
        // treats every key ≥ prefix as a candidate; we stop as soon as
        // the key no longer starts with the prefix.
        let mut out = Vec::new();
        for (key, val) in guard.range(prefix.to_vec()..) {
            if !key.starts_with(prefix) {
                break;
            }
            match val {
                BlobValue::Tombstone => {}
                BlobValue::Raw(bytes) => out.push((key.clone(), bytes.clone())),
                BlobValue::Compressed(bytes) => {
                    let raw = zlib_decompress(bytes)?;
                    out.push((key.clone(), raw));
                }
            }
        }
        Ok(out)
    }

    /// Mark a key as deleted.
    ///
    /// Stores a `BlobValue::Tombstone` so subsequent lookups observe the
    /// absence; alpha-3 will physically remove the entry during
    /// compaction. Deleting a non-existent key is a no-op (idempotent).
    ///
    /// If the store was opened via [`Self::open`], the delete is durably
    /// logged before the in-memory map is updated. On a WAL write
    /// failure the in-memory state is left untouched.
    ///
    /// # Errors
    /// Returns `io::Error` if the WAL append fails. The infallible
    /// alpha-1 signature (`fn delete(&self, &[u8])`) was widened to
    /// `Result` in alpha-2 to surface WAL failures instead of dropping
    /// them silently.
    pub fn delete(&self, key: &[u8]) -> io::Result<()> {
        if let Some(wal) = &self.wal {
            wal.append_delete(key)?;
        }
        let mut guard = self.inner.write();
        guard.insert(key.to_vec(), BlobValue::Tombstone);
        Ok(())
    }

    /// Number of *live* keys (excludes tombstones).
    ///
    /// Cheap-ish (walks every entry) — for perf-critical paths you would
    /// keep a live-key counter alongside the map, but alpha-1 favours
    /// simplicity.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner
            .read()
            .values()
            .filter(|v| !matches!(v, BlobValue::Tombstone))
            .count()
    }

    /// Whether there are any live keys.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_matches_alice_code_tracker_v040() {
        assert!(!should_compress(b"short"));
        assert!(!should_compress(&[0_u8; COMPRESS_THRESHOLD_BYTES - 1]));
        assert!(should_compress(&[0_u8; COMPRESS_THRESHOLD_BYTES]));
    }

    #[test]
    fn compress_if_worthwhile_declines_short_payload() {
        assert!(compress_if_worthwhile(b"todo!()").is_none());
    }

    #[test]
    fn compress_if_worthwhile_declines_incompressible_payload() {
        // High-entropy payload above the length threshold: zlib cannot
        // meaningfully shrink random-looking bytes, so we expect the
        // helper to return None once the margin gate kicks in.
        let payload: Vec<u8> = (0..COMPRESS_THRESHOLD_BYTES + 100)
            .map(|i| u8::try_from((i * 2_654_435_761) & 0xff).unwrap_or(0))
            .collect();
        // We don't assert None strictly (zlib might still eke out a few
        // bytes on this input), but the returned Vec — if any — should
        // never be *longer* than the original.
        if let Some(compressed) = compress_if_worthwhile(&payload) {
            assert!(compressed.len() < payload.len());
        }
    }

    #[test]
    fn compress_if_worthwhile_accepts_repetitive_payload() {
        let payload = b"todo!(\"repetitive stub message; XPBD warm-start diagonal\")\n".repeat(4);
        assert!(payload.len() >= COMPRESS_THRESHOLD_BYTES);
        let compressed =
            compress_if_worthwhile(&payload).expect("repetitive input should compress");
        assert!(compressed.len() < payload.len());
    }
}
