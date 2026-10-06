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

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};
use std::io;
#[cfg(feature = "fs")]
use std::path::{Path, PathBuf};
use std::sync::Arc;

use alice_core::compression::{zlib_compress, zlib_decompress};
use parking_lot::RwLock;

#[cfg(feature = "fs")]
use crate::blob_sstable::{
    enumerate_sstables, max_sstable_seq, remove_stale_tmp_files, sstable_filename_for_seq,
    BlobSstable, FlushMode, LoadedSstable,
};
#[cfg(feature = "fs")]
use crate::blob_wal::{BlobWal, SyncPolicy, WalRecord};

/// Default byte threshold above which a subsequent `put` / `delete`
/// triggers an implicit flush of the WAL into an `SSTable`.
///
/// Chosen to keep the WAL replay cost bounded on reopen while not
/// firing on typical short-burst workloads. Callers who want a
/// different budget open the store through
/// [`BlobStorage::open_with_config`].
#[cfg(feature = "fs")]
pub const DEFAULT_WAL_FLUSH_THRESHOLD_BYTES: u64 = 4 * 1024 * 1024;

/// Default upper bound on the number of `SSTable` files that may
/// accumulate under `FlushMode::Append` before an auto-compaction runs.
#[cfg(feature = "fs")]
pub const DEFAULT_MAX_SSTABLES_BEFORE_COMPACTION: usize = 4;

/// Persistence configuration for the blob store.
///
/// Introduced in v0.2.0-alpha.4 alongside `SSTable`-backed flushes.
/// `flush_mode` / `max_sstables_before_compaction` arrived in
/// v0.2.0-alpha.5 for the multi-`SSTable` path.
#[cfg(feature = "fs")]
#[derive(Debug, Clone, Copy)]
pub struct BlobStorageConfig {
    /// See [`crate::blob_wal::SyncPolicy`].
    pub sync_policy: SyncPolicy,
    /// Once the WAL grows past this many bytes, the next mutation
    /// triggers an automatic flush of the current in-memory state into
    /// a fresh `SSTable` and truncates the WAL. Set to `u64::MAX` to
    /// disable auto-flush entirely.
    pub wal_flush_threshold_bytes: u64,
    /// How flushes interact with existing `SSTable` files. See
    /// [`FlushMode`].
    pub flush_mode: FlushMode,
    /// Under `FlushMode::Append`, an auto-compaction kicks in once the
    /// number of `SSTable` files reaches this value. Ignored for
    /// `FlushMode::Overwrite`. Set to `usize::MAX` to disable
    /// auto-compaction and rely on manual
    /// [`BlobStorage::compact_all_sstables`] calls instead.
    pub max_sstables_before_compaction: usize,
}

#[cfg(feature = "fs")]
impl Default for BlobStorageConfig {
    fn default() -> Self {
        Self {
            sync_policy: SyncPolicy::default(),
            wal_flush_threshold_bytes: DEFAULT_WAL_FLUSH_THRESHOLD_BYTES,
            flush_mode: FlushMode::default(),
            max_sstables_before_compaction: DEFAULT_MAX_SSTABLES_BEFORE_COMPACTION,
        }
    }
}

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
    /// Post-flush live diff: recently-written entries not yet materialised
    /// into an `SSTable`. Reads consult this before falling back to the
    /// on-disk `SSTables`. Tombstones live here to mask keys carried by
    /// older `SSTables` (see the `flush_to_sstable` docs for the tombstone
    /// lifetime story).
    memtable: Arc<RwLock<BTreeMap<Vec<u8>, BlobValue>>>,
    /// On-disk `SSTables` currently loaded via mmap, in **newest-first**
    /// order so `get` can iterate and return as soon as any file
    /// resolves the key.
    ///
    /// The mappings are owned by this list and are only ever read while
    /// holding its read lock (no `Arc` clones escape it). Holding the
    /// write lock therefore means no thread in this process has any of
    /// these files mapped once the list is emptied, which is what lets
    /// compaction rename over and delete them on Windows.
    #[cfg(feature = "fs")]
    sstables: Arc<RwLock<Vec<LoadedSstable>>>,
    /// Serialises `SSTable` publication (append flushes and
    /// compactions) so that two of them never interleave their
    /// rename / delete / WAL-truncate steps.
    #[cfg(feature = "fs")]
    publish_lock: Arc<parking_lot::Mutex<()>>,
    /// Optional durable WAL. `None` means the store is in-memory only.
    #[cfg(feature = "fs")]
    wal: Option<Arc<BlobWal>>,
    /// Path of the canonical single-`SSTable` produced by
    /// `FlushMode::Overwrite`. Retained even under `FlushMode::Append`
    /// so backward-compat reads still work; `Append` also uses this
    /// path's parent directory to place its numbered `SSTables`.
    #[cfg(feature = "fs")]
    sstable_path: Option<PathBuf>,
    /// Auto-flush threshold (see [`BlobStorageConfig`]). Only consulted
    /// on the durable variant.
    #[cfg(feature = "fs")]
    wal_flush_threshold_bytes: u64,
    /// See [`FlushMode`]. Set at open time and immutable afterwards.
    #[cfg(feature = "fs")]
    flush_mode: FlushMode,
    /// See [`BlobStorageConfig::max_sstables_before_compaction`].
    #[cfg(feature = "fs")]
    max_sstables_before_compaction: usize,
    /// Next sequence number to hand out for an append-mode flush.
    /// Wrapped in an `Arc<Mutex>` so cloned `BlobStorage` handles agree
    /// on the counter.
    #[cfg(feature = "fs")]
    next_sstable_seq: Arc<parking_lot::Mutex<u64>>,
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

    /// Open a durable blob store backed by a WAL at `wal_path` using
    /// the default [`SyncPolicy::EveryWrite`] and default flush
    /// threshold.
    ///
    /// The WAL and (optional) sibling `<wal_path stem>.sst` `SSTable` are
    /// created if missing. On load, the `SSTable` is read first to
    /// populate the in-memory map, then the WAL is replayed on top so
    /// that state written after the last flush is visible again. This
    /// makes v0.2.0-alpha.3 databases (which never wrote an `SSTable`)
    /// upgrade transparently: the missing `SSTable` is treated as empty.
    ///
    /// # Errors
    /// Returns `io::Error` if either the WAL or the `SSTable` cannot be
    /// opened, locked, or parsed.
    #[cfg(feature = "fs")]
    pub fn open(wal_path: impl AsRef<Path>) -> io::Result<Self> {
        Self::open_with_config(wal_path, BlobStorageConfig::default())
    }

    /// Deprecated alias kept for α-3.1 callers. Prefer
    /// [`Self::open_with_config`].
    ///
    /// # Errors
    /// See [`Self::open_with_config`].
    #[cfg(feature = "fs")]
    pub fn open_with_policy(
        wal_path: impl AsRef<Path>,
        sync_policy: SyncPolicy,
    ) -> io::Result<Self> {
        Self::open_with_config(
            wal_path,
            BlobStorageConfig {
                sync_policy,
                ..BlobStorageConfig::default()
            },
        )
    }

    /// Open with an explicit [`BlobStorageConfig`].
    ///
    /// # Errors
    /// Returns `io::Error` if the WAL or `SSTable` cannot be opened,
    /// locked, or parsed.
    #[cfg(feature = "fs")]
    pub fn open_with_config(
        wal_path: impl AsRef<Path>,
        config: BlobStorageConfig,
    ) -> io::Result<Self> {
        let wal_path = wal_path.as_ref().to_path_buf();
        let sstable_path = sibling_sstable_path(&wal_path);
        let dir = wal_path
            .parent()
            .map_or_else(|| PathBuf::from("."), std::path::Path::to_path_buf);
        let wal = BlobWal::open_with_policy(&wal_path, config.sync_policy)?;

        // The WAL lock is held from here on, so no other writer can be
        // mid-write in this directory: any `.tmp` left behind belongs to
        // a write that crashed before it was published.
        remove_stale_tmp_files(&dir)?;

        let sstables_newest_first = load_sstables_newest_first(&dir)?;

        // Replay the WAL into the memtable. Reads see WAL state on top of
        // the mmap'd SSTables.
        let mut memtable: BTreeMap<Vec<u8>, BlobValue> = BTreeMap::new();
        for record in wal.replay()? {
            match record {
                WalRecord::Put(key, value) => {
                    memtable.insert(key, value);
                }
                WalRecord::Delete(key) => {
                    memtable.insert(key, BlobValue::Tombstone);
                }
            }
        }

        // Pre-compute the next append-mode sequence number so
        // `flush_to_sstable` need only bump the counter under a lock.
        let next_seq = max_sstable_seq(&dir)?.map_or(1, |s| s.saturating_add(1));

        Ok(Self {
            memtable: Arc::new(RwLock::new(memtable)),
            sstables: Arc::new(RwLock::new(sstables_newest_first)),
            publish_lock: Arc::new(parking_lot::Mutex::new(())),
            wal: Some(Arc::new(wal)),
            sstable_path: Some(sstable_path),
            wal_flush_threshold_bytes: config.wal_flush_threshold_bytes,
            flush_mode: config.flush_mode,
            max_sstables_before_compaction: config.max_sstables_before_compaction,
            next_sstable_seq: Arc::new(parking_lot::Mutex::new(next_seq)),
        })
    }

    /// Force a durable fsync of the WAL and, if the WAL has grown past
    /// the configured threshold, roll the current in-memory state into
    /// a fresh `SSTable` and truncate the WAL.
    ///
    /// The `SSTable` rewrite is skipped when the store is below the
    /// threshold — that path is what `SyncPolicy::EveryWrite` already
    /// keeps durable. Callers who want to force a rewrite unconditionally
    /// should use [`Self::flush_to_sstable`] directly.
    ///
    /// Returns `Ok(())` for in-memory stores created via [`Self::new`].
    ///
    /// # Errors
    /// Propagates the underlying WAL / `SSTable` error.
    pub fn flush(&self) -> io::Result<()> {
        // Without the `fs` feature the store is always the in-memory
        // variant, which has no WAL to sync.
        #[cfg(not(feature = "fs"))]
        return Ok(());
        #[cfg(feature = "fs")]
        let Some(wal) = &self.wal
        else {
            return Ok(());
        };
        #[cfg(feature = "fs")]
        {
            wal.flush()?;

            // Auto-flush if the WAL has grown past the threshold. We ignore
            // the `sstable_path == None` case defensively; a durable store
            // always has one set by `open_with_config`.
            if let Some(_sst_path) = &self.sstable_path {
                let size = wal.size_on_disk()?;
                if size >= self.wal_flush_threshold_bytes {
                    self.flush_to_sstable()?;
                }
            }
            Ok(())
        }
    }

    /// Rewrite the in-memory snapshot into a fresh `SSTable` and truncate
    /// the WAL. Callers can invoke this at will (e.g. at shutdown, or
    /// after a large bulk load) to keep reopen latency low.
    ///
    /// The write is atomic against readers: the new `SSTable` is first
    /// materialised in a sibling `.tmp` path and then renamed into
    /// place. Append mode always writes a fresh `blob-{seq:06}.sst`;
    /// Overwrite mode goes through [`Self::compact_all_sstables`],
    /// which unmaps the old file before replacing it. A crash between
    /// rename and truncate is safe: the WAL replay on next open
    /// produces the same state (`SSTable` + full WAL), the auto-flush
    /// simply re-runs.
    ///
    /// Returns `Ok(())` for in-memory stores created via [`Self::new`].
    ///
    /// # Errors
    /// Propagates any `SSTable` write, WAL truncate, or `SSTable` rename
    /// error.
    #[cfg(feature = "fs")]
    pub fn flush_to_sstable(&self) -> io::Result<()> {
        let Some(sstable_path) = &self.sstable_path else {
            return Ok(());
        };
        let Some(wal) = &self.wal else {
            return Ok(());
        };

        match self.flush_mode {
            FlushMode::Overwrite => {
                // Overwrite mode: fold memtable + every mmap'd SSTable
                // into one new `blob.sst`, replace the sstable list,
                // clear memtable, truncate WAL. Semantically identical
                // to `compact_all_sstables`; we route through the shared
                // helper.
                self.compact_all_sstables()
            }
            FlushMode::Append => {
                // Append mode: the whole memtable — including
                // tombstones — moves into a new sequentially-numbered
                // SSTable so deletes survive a process restart. As of
                // v0.2.0-alpha.8 (`FORMAT_VERSION_V3`), tombstones are
                // a first-class record kind and no longer need to live
                // only in the WAL / memtable.
                let _publish = self.publish_lock.lock();
                let memtable = self.memtable.read();
                let entries: Vec<(Vec<u8>, BlobValue)> = memtable
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                drop(memtable);

                // Reserve the next sequence number under the counter
                // lock so concurrent flushes never collide.
                let dir = sstable_path
                    .parent()
                    .map_or_else(|| PathBuf::from("."), std::path::Path::to_path_buf);
                let target_path = {
                    let mut seq_guard = self.next_sstable_seq.lock();
                    let seq = *seq_guard;
                    *seq_guard = seq.saturating_add(1);
                    dir.join(sstable_filename_for_seq(seq))
                };

                BlobSstable::write_from_iter(
                    &target_path,
                    entries.iter().map(|(k, v)| (k.as_slice(), v)),
                )?;

                // Load the new SSTable and prepend it to the list
                // (newest-first order). The new file has a fresh name, so
                // the rename above never targets a file we have open.
                let new_sst = LoadedSstable::open_mmap(&target_path)?.ok_or_else(|| {
                    io::Error::other("newly-written sstable disappeared before it could be loaded")
                })?;
                {
                    let mut sstables = self.sstables.write();
                    sstables.insert(0, new_sst);
                }

                // Truncate the WAL and clear the memtable — every
                // record now lives in the SSTable. On reopen the WAL
                // replay finds nothing to add and reads route through
                // the sstable path, where tombstones mask stale values
                // from older SSTables just as they did in memory.
                wal.flush()?;
                wal.truncate()?;
                self.memtable.write().clear();
                Ok(())
            }
        }
    }

    /// Merge every `SSTable` currently on disk (plus the in-memory
    /// state) into a single new `SSTable` file, and delete the older
    /// ones. Available on the durable variant only; no-op otherwise.
    ///
    /// This is what `FlushMode::Append` eventually funnels into when
    /// the number of files reaches
    /// [`BlobStorageConfig::max_sstables_before_compaction`]. Callers
    /// can also invoke it manually — e.g. at shutdown — to keep next
    /// reopen fast.
    ///
    /// # Steps
    ///
    /// 1. Write the merged `SSTable` to the `.tmp` sibling of its
    ///    target (`blob.sst` in Overwrite mode, a fresh
    ///    `blob-{seq:06}.sst` in Append mode) and fsync it.
    /// 2. Take the write lock on the `SSTable` list and drop every
    ///    mapping. Readers only touch mappings under the read lock, so
    ///    from here on no thread in this process has any `SSTable`
    ///    open — which Windows requires before a file may be renamed
    ///    over or deleted.
    /// 3. Rename the `.tmp` file onto the target and map it as the only
    ///    entry of the list.
    /// 4. Delete the old files, oldest first, stopping at the first
    ///    failure.
    /// 5. Truncate the WAL and clear the memtable.
    ///
    /// # Crash safety
    ///
    /// Every on-disk state between these steps reopens to the same
    /// contents. Before step 3 the old `SSTables` and the WAL are
    /// untouched and the `.tmp` file is discarded on the next open.
    /// After step 3 the merged file holds the newest record of every
    /// key that any old file held, and the WAL still holds every
    /// memtable record. An old file that has not been deleted yet only
    /// ever holds records that are either superseded by a newer file
    /// that is also still present, or equal to what the merged file
    /// holds: deleting oldest first keeps every remaining old file
    /// newer than every deleted one. The WAL is truncated only after
    /// the last delete succeeded, so tombstones that the merged file
    /// dropped are still replayed while any older value survives on
    /// disk.
    ///
    /// # Errors
    /// Propagates any `SSTable` write, rename, delete, or WAL error. A
    /// failed delete leaves the WAL untouched; the next compaction
    /// retries it.
    ///
    /// # Panics
    /// With the test-only `open-file-audit` feature, panics if a step
    /// would rename over or delete a file that is still mapped.
    #[cfg(feature = "fs")]
    pub fn compact_all_sstables(&self) -> io::Result<()> {
        let Some(sstable_path) = &self.sstable_path else {
            return Ok(());
        };
        let Some(wal) = &self.wal else {
            return Ok(());
        };
        let dir = sstable_path
            .parent()
            .map_or_else(|| PathBuf::from("."), std::path::Path::to_path_buf);
        let _publish = self.publish_lock.lock();

        // Fold everything — old SSTables (oldest → newest) and the
        // memtable — into a single sorted view. Later inserts overwrite
        // earlier ones, so tombstones in the memtable correctly mask any
        // stored values inherited from older SSTables.
        let mut merged: BTreeMap<Vec<u8>, BlobValue> = BTreeMap::new();
        {
            let sstables = self.sstables.read();
            // `sstables` is newest-first; iterate `rev()` to visit
            // oldest first so newer values overwrite.
            for sst in sstables.iter().rev() {
                for (key, value) in sst.iter() {
                    merged.insert(key.to_vec(), value);
                }
            }
        }
        {
            let memtable = self.memtable.read();
            for (key, value) in memtable.iter() {
                merged.insert(key.clone(), value.clone());
            }
        }

        // Filter tombstones for the merged on-disk file. Every old file
        // is deleted before the WAL (which still carries the memtable's
        // deletes) is truncated, so no older value can outlive the
        // tombstone that masked it.
        let entries: Vec<(Vec<u8>, BlobValue)> = merged
            .into_iter()
            .filter(|(_, v)| !matches!(v, BlobValue::Tombstone))
            .collect();

        // Files on disk now, oldest first. These are what step 4 deletes.
        let existing = enumerate_sstables(&dir)?;

        let target_path = match self.flush_mode {
            FlushMode::Overwrite => sstable_path.clone(),
            FlushMode::Append => {
                let mut seq_guard = self.next_sstable_seq.lock();
                let seq = *seq_guard;
                *seq_guard = seq.saturating_add(1);
                drop(seq_guard);
                dir.join(sstable_filename_for_seq(seq))
            }
        };

        // Step 1.
        let tmp_path = BlobSstable::write_tmp_from_iter(
            &target_path,
            entries.iter().map(|(k, v)| (k.as_slice(), v)),
        )?;
        crate::fs_audit::crash_point("compact:after_write")?;

        let mut sstables = self.sstables.write();
        // Step 2. Dropping the list unmaps and closes every old file.
        sstables.clear();
        // Step 3.
        let published = BlobSstable::commit_tmp(&tmp_path, &target_path)
            .and_then(|()| crate::fs_audit::crash_point("compact:after_swap"))
            .and_then(|()| {
                LoadedSstable::open_mmap(&target_path)?.ok_or_else(|| {
                    io::Error::other("compacted sstable disappeared before it could be loaded")
                })
            });
        match published {
            Ok(new_sst) => sstables.push(new_sst),
            Err(e) => {
                // Whatever is on disk is a consistent state (see the
                // crash-safety notes); serve it rather than an empty list.
                *sstables = load_sstables_newest_first(&dir)?;
                return Err(e);
            }
        }

        // Step 4. Oldest first, and stop at the first failure: every
        // remaining old file must stay newer than every deleted one.
        crate::fs_audit::crash_point("compact:before_remove")?;
        for old_path in existing {
            if old_path == target_path {
                continue;
            }
            crate::fs_audit::check_remove(&old_path);
            match std::fs::remove_file(&old_path) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(e) => {
                    return Err(io::Error::new(
                        e.kind(),
                        format!(
                            "compact_all_sstables: could not remove superseded sstable {}: {e}",
                            old_path.display()
                        ),
                    ));
                }
            }
            crate::fs_audit::crash_point("compact:after_first_remove")?;
        }

        // Step 5. The merged SSTable now owns every live value.
        crate::fs_audit::crash_point("compact:before_wal_truncate")?;
        wal.flush()?;
        wal.truncate()?;
        self.memtable.write().clear();
        drop(sstables);
        Ok(())
    }

    /// Number of `SSTable` files currently on disk.
    ///
    /// Exposed for tests and dashboards; `FlushMode::Overwrite` always
    /// reports 0 or 1, `FlushMode::Append` reports the accumulating
    /// count before the next compaction fires.
    ///
    /// # Errors
    /// Propagates the underlying directory scan error.
    #[cfg(feature = "fs")]
    pub fn sstable_count(&self) -> io::Result<usize> {
        // v0.2.0-alpha.7: the authoritative count is the loaded list.
        // We keep the same signature for callers still passing through
        // `io::Result` — an in-memory read cannot fail.
        Ok(self.sstables.read().len())
    }

    /// Whether the durable variant has crossed its auto-flush threshold
    /// and should be re-rolled into an `SSTable`. Public for tests and
    /// diagnostic uses.
    ///
    /// # Errors
    /// Propagates the underlying WAL size query.
    #[cfg(feature = "fs")]
    pub fn wal_needs_flush(&self) -> io::Result<bool> {
        let Some(wal) = &self.wal else {
            return Ok(false);
        };
        Ok(wal.size_on_disk()? >= self.wal_flush_threshold_bytes)
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
        #[cfg(feature = "fs")]
        if let Some(wal) = &self.wal {
            wal.append_put(key, &stored)?;
        }
        {
            let mut guard = self.memtable.write();
            guard.insert(key.to_vec(), stored);
        }
        #[cfg(feature = "fs")]
        self.maybe_auto_flush()?;
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
        // Memtable first: it always carries the freshest state, including
        // tombstones that must mask any older SSTable value.
        {
            let memtable = self.memtable.read();
            if let Some(v) = memtable.get(key) {
                return match v {
                    BlobValue::Tombstone => Ok(None),
                    BlobValue::Raw(bytes) => Ok(Some(bytes.clone())),
                    BlobValue::Compressed(bytes) => {
                        let raw = zlib_decompress(bytes)?;
                        Ok(Some(raw))
                    }
                };
            }
        }
        // SSTable fallback. Snapshot the list (cheap: `Vec<Arc<...>>`)
        // so we don't hold the read lock while probing each file.
        #[cfg(feature = "fs")]
        for sst in self.sstables.read().iter() {
            if let Some(v) = sst.get(key) {
                return match v {
                    BlobValue::Tombstone => Ok(None),
                    BlobValue::Raw(bytes) => Ok(Some(bytes)),
                    BlobValue::Compressed(bytes) => {
                        let raw = zlib_decompress(&bytes)?;
                        Ok(Some(raw))
                    }
                };
            }
        }
        Ok(None)
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
    /// # Panics
    /// Never in production. The internal merge asserts an invariant
    /// (`sources[i].advance()` yields `Some` whenever the heap holds a
    /// reference to `sources[i]`) via `expect`; a violation would mean
    /// the heap grew out of sync with the source vector, which the
    /// implementation forbids by construction.
    pub fn scan_prefix(&self, prefix: &[u8]) -> io::Result<Vec<(Vec<u8>, Vec<u8>)>> {
        // v0.2.0-alpha.8: k-way merge across the memtable and every
        // mmap'd SSTable, priority-ordered so that the newest observation
        // of each key wins. The heap holds one `(key, source_index)`
        // entry per non-empty source; each pop advances the winning
        // source and, if any other source peeks at the same key, drains
        // it too (older duplicates are masked).
        //
        // Priority ordering (lower index = higher priority):
        //   0                → memtable
        //   1 ..= sstables.len() → sstables in newest-first order
        //     (sstables[0] is priority 1, oldest is priority N)
        //
        // Wrapping key and priority in `Reverse` turns the max-heap
        // into a min-heap on `(key, priority)`.
        //
        // The `SSTable` list's read lock is held for the whole scan:
        // mappings must not be used outside it (see the `sstables`
        // field), and holding it across the memtable read below keeps
        // the two views from a single moment (compaction swaps the list
        // and clears the memtable under the list's write lock).
        #[cfg(feature = "fs")]
        let sstables = self.sstables.read();
        #[cfg(feature = "fs")]
        let source_count = 1 + sstables.len();
        #[cfg(not(feature = "fs"))]
        let source_count = 1;

        // Collect each source into a sorted Vec of `(key, value)` pairs
        // upfront. Streaming from the mmap directly across the heap
        // would fight the borrow checker; a materialised Vec keeps the
        // merge simple. The keys are already sorted (memtable via
        // BTreeMap range, sstables via BTreeMap index) so no per-source
        // sort is needed.
        let mut sources: Vec<SortedRun> = Vec::with_capacity(source_count);
        {
            let memtable = self.memtable.read();
            let memtable_entries: Vec<(Vec<u8>, BlobValue)> = memtable
                .range(prefix.to_vec()..)
                .take_while(|(k, _)| k.starts_with(prefix))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            sources.push(SortedRun {
                entries: memtable_entries,
                pos: 0,
            });
        }
        #[cfg(feature = "fs")]
        for sst in sstables.iter() {
            let sst_entries: Vec<(Vec<u8>, BlobValue)> = sst
                .iter_prefix(prefix)
                .map(|(k, v)| (k.to_vec(), v))
                .collect();
            sources.push(SortedRun {
                entries: sst_entries,
                pos: 0,
            });
        }

        // Seed the heap with the first key from each non-empty source.
        let mut heap: BinaryHeap<(Reverse<Vec<u8>>, Reverse<usize>)> = BinaryHeap::new();
        for (idx, source) in sources.iter().enumerate() {
            if let Some(peek) = source.peek() {
                heap.push((Reverse(peek.0.clone()), Reverse(idx)));
            }
        }

        let mut out: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        while let Some((Reverse(winner_key), Reverse(winner_idx))) = heap.pop() {
            // Consume the winning source's entry.
            let winner_value = sources[winner_idx]
                .advance()
                .expect("heap entry must reference a live source position")
                .1;
            if let Some(next) = sources[winner_idx].peek() {
                heap.push((Reverse(next.0.clone()), Reverse(winner_idx)));
            }

            // Drain any other sources peeking at the same key — they
            // are older observations that the winner masks.
            while let Some(&(Reverse(ref top_key), Reverse(top_idx))) = heap.peek() {
                if top_key != &winner_key {
                    break;
                }
                heap.pop();
                sources[top_idx].advance();
                if let Some(next) = sources[top_idx].peek() {
                    heap.push((Reverse(next.0.clone()), Reverse(top_idx)));
                }
            }

            // Emit the winner if it is a live value.
            match winner_value {
                BlobValue::Tombstone => {}
                BlobValue::Raw(bytes) => out.push((winner_key, bytes)),
                BlobValue::Compressed(bytes) => {
                    let raw = zlib_decompress(&bytes)?;
                    out.push((winner_key, raw));
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
        #[cfg(feature = "fs")]
        if let Some(wal) = &self.wal {
            wal.append_delete(key)?;
        }
        {
            let mut guard = self.memtable.write();
            guard.insert(key.to_vec(), BlobValue::Tombstone);
        }
        #[cfg(feature = "fs")]
        self.maybe_auto_flush()?;
        Ok(())
    }

    /// If the WAL has grown past the configured threshold, roll the
    /// current in-memory state into a fresh `SSTable` and truncate the
    /// WAL. Called at the tail of every successful `put` / `delete`.
    ///
    /// Cheap when the WAL is small: a single `metadata` syscall. A
    /// pathological workload that hovers just under the threshold
    /// pays for that syscall on every write; α-3.3 introduces a
    /// cheaper heuristic (per-record counter) alongside compaction.
    #[cfg(feature = "fs")]
    fn maybe_auto_flush(&self) -> io::Result<()> {
        if self.wal_needs_flush()? {
            self.flush_to_sstable()?;
        }
        // Under FlushMode::Append the per-flush cost is O(delta) but
        // SSTable files accumulate; roll them back into one when the
        // count reaches the configured cap.
        if self.flush_mode == FlushMode::Append
            && self.sstable_count()? >= self.max_sstables_before_compaction
        {
            self.compact_all_sstables()?;
        }
        Ok(())
    }

    /// Number of *live* keys (excludes tombstones).
    ///
    /// Walks every entry — memtable keys and each mmap'd `SSTable`'s
    /// offset index — resolving newest-wins. For perf-critical paths a
    /// live-key counter would be cheaper, but the alpha semantics
    /// favour simplicity and honest tombstone handling.
    #[must_use]
    pub fn len(&self) -> usize {
        // Reuse the scan_prefix merge logic, but stay in-memory:
        // materialising values is what makes scan_prefix expensive, so
        // do the merge without decompressing.
        let mut seen: BTreeMap<Vec<u8>, bool> = BTreeMap::new(); // key → is_live
        #[cfg(feature = "fs")]
        let sstables = self.sstables.read();
        #[cfg(feature = "fs")]
        for sst in sstables.iter().rev() {
            for (key, value) in sst.iter() {
                seen.insert(key.to_vec(), !matches!(value, BlobValue::Tombstone));
            }
        }
        {
            let memtable = self.memtable.read();
            for (key, value) in memtable.iter() {
                seen.insert(key.clone(), !matches!(value, BlobValue::Tombstone));
            }
        }
        seen.values().filter(|live| **live).count()
    }

    /// Whether there are any live keys.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Map every `SSTable` in `dir`, newest first (read-path priority).
#[cfg(feature = "fs")]
fn load_sstables_newest_first(dir: &Path) -> io::Result<Vec<LoadedSstable>> {
    // `enumerate_sstables` returns oldest → newest by sequence.
    let mut out = Vec::new();
    for sst_path in enumerate_sstables(dir)? {
        if let Some(sst) = LoadedSstable::open_mmap(&sst_path)? {
            out.push(sst);
        }
    }
    out.reverse();
    Ok(out)
}

/// Derive the `SSTable` path from the WAL path.
///
/// The convention is `blob.wal` -> `blob.sst`. Any other stem is
/// preserved verbatim so a caller passing `foo.wal` sees `foo.sst`.
#[cfg(feature = "fs")]
fn sibling_sstable_path(wal_path: &Path) -> PathBuf {
    let mut sst = wal_path.to_path_buf();
    let file_stem = sst
        .file_stem()
        .map(std::ffi::OsStr::to_os_string)
        .unwrap_or_default();
    let mut with_ext = file_stem;
    with_ext.push(".sst");
    sst.set_file_name(with_ext);
    sst
}

/// A pre-materialised sorted sequence used as one arm of the k-way
/// merge in [`BlobStorage::scan_prefix`]. Introduced in
/// v0.2.0-alpha.8. Kept internal because the merge is the only caller.
struct SortedRun {
    entries: Vec<(Vec<u8>, BlobValue)>,
    pos: usize,
}

impl SortedRun {
    fn peek(&self) -> Option<&(Vec<u8>, BlobValue)> {
        self.entries.get(self.pos)
    }

    fn advance(&mut self) -> Option<(Vec<u8>, BlobValue)> {
        let out = self.entries.get(self.pos).cloned();
        if out.is_some() {
            self.pos += 1;
        }
        out
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
