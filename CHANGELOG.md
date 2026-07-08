# Changelog

All notable changes to ALICE-DB will be documented in this file.

## [0.2.0-alpha.7] - 2026-07-08

α-3.3b-2 subset: on-disk read fast path via memory-mapped SSTables and the `BlobStorage` architectural split.

`BlobStorage` now separates its state into a `memtable` (recent writes replayed from the WAL) and a newest-first `Vec<Arc<LoadedSstable>>` (settled state, held via mmap). Reads consult the memtable first, then fall through to each SSTable in order, using the Bloom filter shipped in v0.2.0-alpha.6 to reject absent keys before touching the mmap. Only key offsets stay in memory; the value bytes remain on disk until a read actually copies them out. Existing databases (α-3.2a onwards) open unchanged.

### Added

- `blob_sstable::LoadedSstable` — mmap-backed `SSTable` reader.
  - `open_mmap(path)` — walks the records section once to verify per-record CRCs and build the key → offset index. Missing files return `Ok(None)` so a fresh directory opens cleanly.
  - `get(key) -> Option<BlobValue>` — Bloom-guarded point lookup; returns `None` without touching the index when the Bloom rejects. On a hit, copies the value bytes out of the mmap into an owned `BlobValue`.
  - `iter()` / `iter_prefix(prefix)` — ordered walks used by compaction and `scan_blob_prefix`.
  - `bloom()` / `path()` / `len()` / `is_empty()` accessors.
  - `Debug` impl surfaces the summary (path, mmap length, Bloom presence, record count) without dumping the raw bytes.
- `blob_sstable::build_offset_index` — internal helper that walks the records slice and returns a `BTreeMap<Vec<u8>, RecordOffset>` (private).
- 6 unit tests in `src/blob_sstable.rs::tests` covering: raw value round-trip, compressed value round-trip verbatim, absent-key resolution, full-range iteration, prefix-range iteration, legacy v1 open path (hand-written to alpha.5 bytes).
- 8 integration tests in `tests/blob_mmap_read.rs`:
  - Flush + reopen serves both `Raw` and `Compressed` values through the mmap path.
  - `FlushMode::Append` multi-`SSTable` reads return newest-wins across two flushes.
  - A memtable tombstone masks a live value in an older `SSTable`, both live and after reopen (WAL replay).
  - Bloom-guarded absent keys return `None` (functional coverage; a perf benchmark is out of scope for α-3.3b-2).
  - `scan_blob_prefix` merges memtable and multiple `SSTables` with newest-wins semantics and stable ascending order.
  - `compact_all_blob_sstables` after four `Append` flushes folds every live key into one file, all values preserved.
  - 8-thread concurrent read/write against a shared store observes a consistent view.
- `memmap2 = "0.9"` dependency (was already declared for a future migration; now actively used).

### Changed

- `BlobStorage` fields:
  - `inner: Arc<RwLock<BTreeMap<Vec<u8>, BlobValue>>>` (which held every live value in memory) is replaced by
  - `memtable: Arc<RwLock<BTreeMap<Vec<u8>, BlobValue>>>` (WAL-replayed diff since the last flush) and
  - `sstables: Arc<RwLock<Vec<Arc<LoadedSstable>>>>` (newest-first).
- `BlobStorage::open_with_config`: enumerates every `SSTable` on disk, opens each via `LoadedSstable::open_mmap`, and reverses the list to newest-first. The WAL replay populates only the memtable — record values living inside `SSTables` stay on disk until a read touches them.
- `BlobStorage::get`: memtable first, then iterates `sstables` newest-first calling `LoadedSstable::get`. Snapshots the sstable list (a cheap `Vec<Arc<_>>` clone) so the read lock does not span the probes.
- `BlobStorage::put` / `BlobStorage::delete`: mutate the memtable only. The WAL still records every change so a reopen reconstructs the same memtable state.
- `BlobStorage::flush_to_sstable`:
  - `FlushMode::Overwrite`: delegates to `compact_all_sstables` — the merged file replaces every prior `SSTable` and the memtable clears.
  - `FlushMode::Append`: writes the memtable's non-tombstone entries to a fresh sequentially-numbered `SSTable` file, mmap-opens it, prepends it to the sstable list, prunes tombstones from the memtable, and truncates the WAL. Semantically identical to v0.2.0-alpha.5.
- `BlobStorage::compact_all_sstables`: folds every mmap'd `SSTable` (walking `sstables` in `rev()` for oldest-first order) plus the memtable into a merged `BTreeMap`. Tombstones drop out safely because we rewrite every file in one operation. The new file is written to a `.tmp` sibling, atomically renamed into place, mmap-opened, and swapped into `sstables` **before** the older files are unlinked. On Unix an unlink under a live mmap is safe; on Windows the deletion may fail and is logged rather than propagated, leaving stragglers to be absorbed by the next compaction.
- `BlobStorage::scan_prefix`: merges memtable and every `SSTable`'s prefix window into a `BTreeMap` accumulator with newest-wins semantics (`sstables.iter().rev()` for oldest-first, then memtable on top). Materialises values (decompressing where needed) and skips tombstones.
- `BlobStorage::len`: reuses the merge approach — walks every source without decompressing and counts non-tombstoned survivors.
- `BlobStorage::sstable_count`: now reports the length of the loaded list (`self.sstables.read().len()`) rather than a directory scan; the signature retains `io::Result` so callers do not need to change.

### Backward compatibility

- Existing databases (v0.2.0-alpha.2 through v0.2.0-alpha.6) open unchanged. `LoadedSstable::open_mmap` accepts both format v1 and v2 files.
- `FlushMode::Overwrite` (the default) preserves v0.2.0-alpha.5 semantics exactly: every flush rewrites `blob.sst` and clears the memtable.
- `FlushMode::Append` preserves v0.2.0-alpha.5 semantics including the pre-existing limitation that a delete-then-flush cycle cannot survive a process restart (see below). No new tombstone bug is introduced.
- Public API (`AliceDB::open` / `open_with_blob_config` / `get_blob` / `put_blob` / `delete_blob` / `scan_blob_prefix` / `flush_blobs` / `compact_blob_sstable` / `compact_all_blob_sstables`) is source-compatible.

### Known limitation carried forward

`FlushMode::Append` cannot persist tombstones across a process restart: the `SSTable` format has no tombstone record, so a `delete_blob` followed by a flush and then a reopen may resurrect the value from an older `SSTable`. Same behaviour as v0.2.0-alpha.5. Workarounds:

- Call `compact_all_blob_sstables` after any batch of deletes under `Append`; the merged file drops tombstoned keys.
- Switch to `FlushMode::Overwrite` (the default) if durable deletes are required.

α-3.3b-3 lifts this limitation by extending the `SSTable` format to carry tombstone records.

### Design decisions

- **Key offsets in memory, values on disk**: the `LoadedSstable` index holds one entry per record (`BTreeMap<Vec<u8>, RecordOffset>`), where `RecordOffset` is 32 bytes. Value bytes stay in the mmap and are copied into an owned `BlobValue` only when a read materialises them. For workloads with large values (e.g. compressed `Stub` JSON in ALICE-CodeTracker) this is the significant memory savings.
- **Newest-first sstable list**: reads walk the list in stored order and return as soon as any `SSTable` resolves the key. Compaction visits the list in `rev()` for merge semantics. Keeping newest-first matches the read-path priority and avoids repeated reversals.
- **Bloom-guarded lookup at the SSTable layer**: `LoadedSstable::get` probes the Bloom first (when present) before touching the index. For v1 files (`bloom() == None`) the Bloom is treated as always-accept — correctness is preserved, no acceleration.
- **`Arc<LoadedSstable>` for cheap sharing**: `get` clones the sstable list under the read lock and drops the lock before probing. Cloning a `Vec<Arc<_>>` is O(N) reference bumps, cheap enough for the alpha and future concurrency stories.
- **Swap-then-delete for compaction**: the merged file is materialised and swapped into `sstables` before old files are unlinked. On Unix this is safe against live mmaps (the inode outlives the directory entry). On Windows a still-mapped file cannot be deleted; we log and continue rather than fail the whole compaction, and the next compaction absorbs the leftovers.
- **`scan_prefix` accumulates into a `BTreeMap`**: correct newest-wins semantics with tombstone masking, at the cost of O(N log N) merging. α-3.3b-3 introduces a sorted-run merge iterator that avoids the accumulator.

### α-3 roadmap remainder

- **α-3.3b-3**: sparse index / prefix trie inside each `SSTable` for accelerated `scan_blob_prefix` + tombstone-carrying `SSTable` format extension to lift the `Append`-mode delete limitation.

## [0.2.0-alpha.6] - 2026-07-08

α-3.3b-1 subset: durable investment for the on-disk read fast path. Introduces `SSTable` format v2 with an embedded Bloom filter per file and a longer 24-byte footer (`records_size` + `bloom_size` + magic). Format v1 files written by v0.2.0-alpha.2 through v0.2.0-alpha.5 continue to open and are transparently upgraded to v2 on the next flush or compaction. The read path itself continues to load records into memory — mmap + Bloom-guarded point lookups land in α-3.3b-2.

### Added

- `src/bloom.rs` — dependency-free Bloom filter.
  - `BloomFilter::with_capacity(expected_elements, false_positive_rate)` sized via `m = ⌈-n·ln(p)/(ln 2)²⌉`, `k = ⌈m/n·ln 2⌉`.
  - `BloomFilter::from_raw(bits, num_bits, num_hashes)` reconstructs a filter from its serialised components (used by the SSTable reader).
  - `insert` / `contains` implemented via Kirsch-Mitzenmacher double hashing over `std::hash::DefaultHasher` (SipHash-2-4), so the crate remains dependency-free.
  - 5 unit tests: empty filter rejects, no false negatives, false-positive rate stays within 2× of target at n = 4 096 keys, roundtrip through raw bits preserves membership, sizing scales with element count.
- `blob_sstable::BlobSstable::bloom()` accessor — returns `Some(&BloomFilter)` for v2 files and `None` for legacy v1 files.
- `blob_sstable` constants: `FORMAT_VERSION_V1` / `FORMAT_VERSION_V2` / `CURRENT_VERSION` / `FOOTER_LEN_V1` / `FOOTER_LEN_V2` / `BLOOM_HEADER_LEN` / `BLOOM_FALSE_POSITIVE_RATE`.
- 4 integration tests in `tests/blob_sstable_bloom.rs`:
  - New v2 writes carry a Bloom filter that accepts every inserted key (no false negatives).
  - Bloom rejects clearly-absent keys (≥ 100 / 128 unrelated probes rejected).
  - Legacy v1 files (hand-written to alpha.5 bytes) still open; `bloom()` reports `None`; records read back correctly.
  - `AliceDB::open` on a directory with a legacy v1 `blob.sst` reads it transparently; the first `compact_blob_sstable()` upgrades it to v2.

### Changed

- `BlobSstable::write_from_iter` now:
  - Builds a Bloom filter during record iteration (sized by observed record count at the classic 1 % false-positive rate).
  - Writes the Bloom section between the records and the footer.
  - Emits a v2 footer: `records_size (u64) + bloom_size (u64) + magic (8)` — 24 bytes instead of the v1 16.
  - Header now stamps `FORMAT_VERSION_V2`.
- `BlobSstable::open` branches on the header version:
  - v1: reads the 16-byte footer as `records_size + magic`; `bloom()` returns `None`.
  - v2: reads the 24-byte footer as `records_size + bloom_size + magic`; parses the Bloom section and stores the reconstructed filter.
- Any newly-written or newly-compacted SSTable is v2. Read-side callers that need to distinguish should probe `.bloom().is_some()`.

### Backward compatibility

- v1 files (`ALICEBBS` header + 16-byte footer) written by any of v0.2.0-alpha.2 through v0.2.0-alpha.5 continue to open on `BlobStorage::open_with_config` and `AliceDB::open`. The reader simply reports `bloom() == None` — the read path already probes the record map, so a missing Bloom is a semantic no-op (equivalent to a filter that never rejects).
- The first flush or compaction on such a database rewrites the file as v2 with a Bloom filter attached. No user-visible action is required.

### Design decisions

- **Bloom sized by observed record count**: the writer knows the exact record count once it has iterated the input, so it sizes the filter for that count at the fixed 1 % target rate. This keeps every file's on-disk overhead proportional to its own contents (~1.2 bytes/key).
- **`std::hash::DefaultHasher` for hashing**: SipHash-2-4 is the standard library's built-in hasher, so we avoid adding a hashing dependency at this stage. `α-3.3b-2` can swap in `xxhash-rust` or `wyhash` if benchmarks justify it.
- **Kirsch-Mitzenmacher double hashing**: one SipHash call per key produces two `u64` values (via a differentiating suffix on the second call); the `k` bit positions are derived as `h1 + i · h2 mod m`. This preserves the standard false-positive-rate analysis while doing only one hash call per operation.
- **CRC32C on the Bloom section**: same integrity story as records. Corrupted files fail loudly on open.
- **Read path unchanged in α-3.3b-1**: the Bloom filter is *loaded and available* but not yet consulted by `get_blob`. That gates on the mmap + `LoadedSstable` split that lands in α-3.3b-2; shipping the format now means α-3.3b-2 does not need a second format migration.

### α-3 roadmap remainder

- **α-3.3b-2**: mmap-backed read path (`LoadedSstable` = `memtable` + `Vec<mmap SSTables>`) + Bloom-guarded point lookups on `get_blob`. The format v2 infrastructure shipped here is a prerequisite.
- **α-3.3b-3**: sparse index / prefix trie inside each SSTable for accelerated `scan_blob_prefix`.

## [0.2.0-alpha.5] - 2026-07-08

α-3.3a subset: multi-SSTable path with sequential numbering and simple full-merge compaction. Query-side acceleration (Bloom filter / prefix trie) remains scheduled for α-3.3b. `FlushMode::Overwrite` — matching v0.2.0-alpha.4 semantics — stays the default so pre-α-3.3 callers observe no behaviour change.

### Added

- `blob_sstable::FlushMode` enum (`Copy` + `Default = Overwrite`).
- `blob_sstable::parse_sstable_seq(name)` / `sstable_filename_for_seq(seq)` / `enumerate_sstables(dir)` / `max_sstable_seq(dir)` — helpers that make the sequentially numbered blob SSTable filenames (`blob-{seq:06}.sst`) explicit. `blob.sst` is treated as sequence 0 so pre-α-3.3 databases upgrade in place without a migration step.
- `blob::BlobStorageConfig::flush_mode` (default `Overwrite`) and `blob::BlobStorageConfig::max_sstables_before_compaction` (default 4).
- `blob::DEFAULT_MAX_SSTABLES_BEFORE_COMPACTION` constant.
- `blob::BlobStorage::compact_all_sstables()` — merge every existing SSTable + the in-memory state into one new SSTable and delete the older ones.
- `blob::BlobStorage::sstable_count()` — diagnostic accessor.
- `AliceDB::compact_all_blob_sstables()` and `AliceDB::blob_sstable_count()` mirroring the storage-layer additions.
- 9 integration tests in `tests/blob_multi_sstable.rs`:
  - Filename helper round-trip (`0`, `1`, `42`, `999_999`, `1_000_000`, `u64::MAX`).
  - `blob.sst` recognised as sequence 0; unrelated filenames rejected.
  - `FlushMode::Append` produces sequentially numbered files and leaves prior ones alone.
  - Reopen after multi-flush honours last-write-wins across sequence order.
  - Manual `compact_all_sstables` collapses to one file, all keys preserved.
  - Auto-compaction fires when SSTable count reaches `max_sstables_before_compaction`.
  - Pre-α-3.3 `blob.sst` co-exists with `blob-000001.sst` under `FlushMode::Append`.
  - Corrupted sequential SSTable surfaces on open (fail-loud).

### Changed

- `BlobStorage::open_with_config` now loads every SSTable file present in the WAL's parent directory (in ascending sequence order) before replaying the WAL. Under the default `FlushMode::Overwrite` this reduces to the α-3.2a behaviour (only `blob.sst` is present).
- `BlobStorage::flush_to_sstable` branches on `flush_mode`: `Overwrite` continues to rewrite `blob.sst` in place; `Append` reserves the next sequence number and writes `blob-{seq:06}.sst`, leaving prior SSTables untouched.
- `BlobStorage`'s auto-flush tail on `put` / `delete` now additionally triggers `compact_all_sstables` once `sstable_count() >= max_sstables_before_compaction` (Append only).
- `BlobStorageConfig` gained two new fields; existing struct-literal callers must switch to `..BlobStorageConfig::default()` or provide the new fields explicitly.

### Backward compatibility

- Pre-α-3.3 databases (single `blob.sst` file) open unchanged under either flush mode:
  - `FlushMode::Overwrite` (default): every flush continues to rewrite `blob.sst` — behaviour identical to v0.2.0-alpha.4.
  - `FlushMode::Append`: the legacy `blob.sst` is treated as sequence 0 and preserved until the next `compact_all_sstables` folds it into the merged file. First flush produces `blob-000001.sst` alongside it.
- `AliceDB::open` and `AliceDB::open_with_blob_sync_policy` inherit the `Overwrite` default via `BlobStorageConfig::default()`, so callers that never opt into Append see no behaviour change.

### Design decisions

- **Single unified in-memory `BTreeMap` retained**: α-3.3a keeps the whole live state in memory just like α-3.2a, so multi-SSTable is currently just an on-disk layout change. The real benefits (mmap-backed reads, bloom-guarded point lookups) land in α-3.3b.
- **Directory scan over manifest**: with at most `max_sstables_before_compaction` files at any time, a directory scan on open is O(N) tiny and there is no manifest to keep consistent with disk state.
- **Crash-safety ordering**: the merged SSTable is renamed into place first; only then are the older SSTables deleted one by one. A crash between the rename and the deletes leaves an idempotent state — the next open sees the merged SSTable plus one or more redundant older ones, and the next compaction folds them in.
- **Fail-loud on any SSTable corruption**: same policy as α-3.2a. Even a single corrupted file bubbles out of `AliceDB::open` as `InvalidData`.

### α-3 roadmap remainder

- **α-3.3b**: on-disk read path (mmap) + Bloom filter per SSTable (point lookup fast path) + prefix trie or sparse index (scan acceleration).

## [0.2.0-alpha.4] - 2026-07-08

α-3.2a subset: single blob SSTable + WAL rotation. Multi-SSTable, compaction, and read-side query acceleration (Bloom filter / prefix trie) remain scheduled for α-3.3.

### Added

- `blob_sstable` module — sorted, immutable snapshot of the blob key-value store on disk.
  - `BlobSstable::write_from_iter(path, iter)` atomically writes a sorted, tombstone-free snapshot via `<path>.tmp` + `rename`.
  - `BlobSstable::open(path)` verifies header + footer magic + per-record CRC32C; returns `Ok(None)` for missing files (so pre-α-3.2a DBs upgrade cleanly).
  - File format: `ALICEBBS` header + records + `ALICEEND` footer. See `src/blob_sstable.rs` for the layout table.
- `blob::BlobStorageConfig { sync_policy, wal_flush_threshold_bytes }` — persistence configuration bundle.
- `blob::DEFAULT_WAL_FLUSH_THRESHOLD_BYTES` (4 MiB).
- `BlobStorage::open_with_config(path, config)` — the new canonical open. `open_with_policy` is retained as a thin shim that supplies the default threshold.
- `BlobStorage::flush_to_sstable()` — force a rewrite of the current in-memory snapshot into the SSTable and truncate the WAL.
- `BlobStorage::wal_needs_flush()` — probe the auto-flush threshold; used internally after every `put` / `delete`.
- `BlobWal::size_on_disk()` / `BlobWal::truncate()` — the primitives that let `flush_to_sstable` reset the WAL after a successful SSTable write.
- `AliceDB::open_with_blob_config(path, config)` — surface the full `BlobStorageConfig` at the top-level API.
- `AliceDB::compact_blob_sstable()` — user-facing wrapper for `BlobStorage::flush_to_sstable`.
- 8 unit tests in `src/blob_sstable.rs::tests` (round-trip empty / two raw / compressed / tombstone reject / missing file / corrupted footer / corrupted CRC / size probe).
- 6 integration tests in `tests/blob_sstable_integration.rs` (manual flush + WAL truncate / reopen after flush / SSTable + WAL replay / auto-flush threshold / α-3.3 WAL-only DB upgrade / corrupted SSTable detected on open).

### Changed

- `AliceDB::open` and `AliceDB::open_with_blob_sync_policy` are now thin wrappers over `AliceDB::open_with_blob_config`. The default behaviour is unchanged: `SyncPolicy::EveryWrite` + 4 MiB auto-flush threshold.
- `BlobStorage::put` and `BlobStorage::delete` invoke `maybe_auto_flush` at their tail. When the WAL crosses the configured threshold the current in-memory snapshot is rewritten into the SSTable and the WAL is truncated. Cost is one `metadata` syscall per successful mutation while below the threshold.
- `AliceDB::flush_blobs` may now trigger an SSTable rewrite in addition to the WAL fsync when the threshold has been crossed.

### Backward compatibility

Databases written by v0.2.0-alpha.3 have a `blob.wal` but no `blob.sst`. Alpha-3.2a treats a missing SSTable file as an empty snapshot, so opening such a database succeeds and the WAL alone drives the load. The first `compact_blob_sstable` (or auto-flush) materialises the SSTable in place. No manual migration is required.

### Design decisions

- **Single SSTable per store**: α-3.2a keeps one SSTable per `BlobStorage`. Every flush is effectively a full compaction (tombstones drop out, values are rewritten sorted). Cheap for the ALICE-CodeTracker workload (thousands of blobs, not millions); α-3.3 replaces this with a real LSM (multi-level SSTables + incremental compaction).
- **Atomic rewrite via rename**: `write_from_iter` writes to a sibling `.tmp` path and renames over the destination. On POSIX (Linux, macOS) `rename(2)` is atomic within a filesystem, so a reader never observes a partially written SSTable. A crash between rename and WAL truncate is safe: reopen re-applies the WAL on top of the (already-current) SSTable, and the next auto-flush truncates the WAL again.
- **No manifest**: with only one SSTable per store there is nothing to enumerate. α-3.3 will introduce a manifest when multi-SSTable landing arrives.
- **Fail-loud on SSTable corruption**: a corrupted SSTable footer or record CRC bubbles out of `AliceDB::open` as `InvalidData`. Silent fallback to WAL-only load was considered but rejected because it would mask a serious integrity failure from the dogfooding consumer.

### α-3 roadmap remainder

- **α-3.3**: Multi-SSTable levels + incremental compaction + Bloom filter (point lookup fast path) + prefix trie (scan acceleration).

## [0.2.0-alpha.3] - 2026-07-08

Hardens the blob WAL shipped in alpha-2 with cross-process safety and a configurable durability / throughput trade-off. Alpha-3.1 subset — SSTable format v2 and query acceleration remain scheduled for alpha-3.2 / alpha-3.3.

### Added

- `blob_wal::SyncPolicy` enum (`Copy` + `Default = EveryWrite`):
  - `EveryWrite` — `sync_data` after every append (alpha-2 default, unchanged).
  - `Batched { max_pending_ops }` — buffer up to `max_pending_ops` records, fsync once the threshold is reached. Callers can force an earlier sync with `flush`.
  - `Manual` — never auto-fsync; caller is fully responsible for durability via `flush`.
- `BlobWal::open_with_policy(path, sync_policy)` and `BlobWal::flush()`.
- `blob::BlobStorage::open_with_policy(path, sync_policy)` and `BlobStorage::flush()`.
- `AliceDB::open_with_blob_sync_policy(path, sync_policy)` /
  `AliceDB::with_config_and_blob_sync_policy(config, sync_policy)` /
  `AliceDB::flush_blobs()`.
- Exclusive `fs2::FileExt::try_lock_exclusive` advisory lock on the blob WAL file, taken at open time. A second in-process or cross-process `AliceDB::open` on the same directory returns an `io::Error` with `ErrorKind::WouldBlock` until the first handle is dropped. Locks are released in `BlobWal::Drop` (and implicitly at handle close).
- `Drop for BlobWal` that best-effort fsyncs any pending writes and releases the advisory lock.
- 6 integration tests in `tests/blob_locking_and_sync.rs`:
  - `second_open_on_same_path_fails_while_first_is_alive` — asserts `WouldBlock`.
  - `second_open_succeeds_after_first_is_dropped` — verifies clean re-lock.
  - `batched_policy_persists_after_threshold_is_crossed` (8 puts against `max_pending_ops = 4`).
  - `batched_policy_uses_flush_to_persist_partial_batch`.
  - `manual_policy_persists_only_via_explicit_flush` (10 puts, one final `flush_blobs`).
  - `every_write_policy_is_the_default_and_flush_is_a_no_op`.

### Changed

- `BlobWal` internally moved the file handle and pending-write counter into a `WalState` struct behind a single `parking_lot::Mutex`. This is an internal refactor with no external API change beyond the additions above.
- `pub mod blob_wal` is now re-exported at the crate root (implicit through `use alice_db::blob_wal::SyncPolicy`).

### Design decisions

- **Advisory lock, not mandatory**: `fs2::FileExt::try_lock_exclusive` cooperates with well-behaved writers but does not protect against writers that ignore the lock. Alpha-3 assumes the tracker/consumer opens through `AliceDB::open`, which always takes the lock.
- **Count-based batching only**: alpha-3.1 defers time-based batching (e.g. "fsync every 100 ms") to alpha-3.2 so we do not spawn a background thread yet. `Batched { max_pending_ops }` is simple, deterministic, and covers the ALICE-CodeTracker high-throughput scan case (scan of ALICE-* mono-repo tops out around a few hundred writes per second).
- **Drop-time flush is best-effort**: a panicking process may still lose the last partial batch. Callers who need airtight durability at shutdown should call `flush_blobs` explicitly before the handle drops.

### Alpha-3 roadmap remainder

- **α-3.2**: SSTable format v2, MemTable flush, compaction.
- **α-3.3**: Bloom filter and prefix trie for `scan_blob_prefix` acceleration.

### Downstream

The exclusive lock closes the multi-writer safety gap flagged in alpha-2 and clears the last blocking dependency for **ALICE-CodeTracker v0.5.0**'s `alice-tracker-alicedb` backend design. Compaction is not strictly required for that consumer; it can adopt alpha-3.1 today and pull in later alpha-3.x point releases as the WAL grows unbounded.

## [0.2.0-alpha.2] - 2026-07-08

Adds write-ahead log (WAL) persistence to the blob key-value store shipped in alpha-1. Blob state now survives process restarts: on `AliceDB::open` the WAL is replayed and the in-memory `BTreeMap` is rebuilt from scratch. The time-series engine and blob store use independent files (`data.alice` / `wal.alice` for time-series, `blob.wal` for blobs) so neither disturbs the other.

### Added

- `blob_wal` module — `BlobWal` (append-only file, `parking_lot::Mutex`-guarded), `WalRecord` (`Put` / `Delete`), 10-byte header + variable key/value + 4-byte CRC32C trailer per record.
- `BlobStorage::open(path)` — new constructor that opens (or creates) the WAL at `path`, replays it, and reconstructs the in-memory map. The existing `BlobStorage::new()` remains for ephemeral / test use.
- `AliceDB::open` and `AliceDB::with_config` now place the blob WAL at `<data_dir>/blob.wal` and replay it on open.
- `crc32fast = "1.4"` workspace dependency (fast pure-Rust CRC32C for record checksums).
- 5 unit tests in `src/blob_wal.rs::tests` covering the frame format (roundtrip, tombstone-guard, empty, truncated tail, corrupted checksum).
- 9 integration tests in `tests/blob_persistence.rs` — put/reopen for short and compressed values, delete/reopen, overwrite/reopen, 1000 mixed puts and deletes across restart (5.55 s wall-clock at 1 fsync per op), empty-directory open, truncated WAL tail recovery, corrupted record stops replay, missing WAL file is created cleanly.

### Changed — breaking (still alpha, no compat guarantee)

- `AliceDB::delete_blob(&self, &[u8])` now returns `io::Result<()>` instead of `()`. The WAL append can fail (e.g. disk full) and swallowing that error would silently lose the deletion. Same widening applied to `BlobStorage::delete`. Callers that previously relied on the infallible signature must add `?` or `.unwrap()`.

### Design decisions

- **1 fsync per op**: alpha-2 syncs the file after every append. This keeps the crash-recovery contract simple ("everything replayed is durable, nothing else is") at the cost of throughput. Batched fsync (group commit) is deferred to alpha-3.
- **Corrupted-record semantics**: when replay hits a CRC mismatch we stop reading further records, on the theory that if one payload is corrupted the file's framing offsets may no longer be trustworthy. Truncated tails (short read past a record boundary) are treated as clean end-of-log — the common crash-mid-write case.
- **Single-writer per directory**: alpha-2 does not take a cross-process advisory lock. Two processes opening the same data directory would corrupt the WAL. Documented as an alpha-2 limitation; file locking arrives in alpha-3.
- **No re-encoding on the WAL**: `BlobValue::Compressed(bytes)` is written to the WAL byte-for-byte, so a value is compressed exactly once (on the initial put) even after N reopens.

### Downstream

- **ALICE-CodeTracker v0.5.0** — the durability floor cleared here is the prerequisite for the `alice-tracker-alicedb` backend planned in `~/ALICE-CodeTracker/CHANGELOG.md#040`. Once alpha-2 stabilises we can wire the tracker against it.

## [0.2.0-alpha.1] - 2026-07-08

First alpha of the v0.2.0 line, opening the general-purpose LSM-KV path proposed in `docs/EXPANSION_PROPOSAL.md`. The time-series API is unchanged; the new blob store sits alongside it and is opt-in through a distinct set of methods.

### Added

- `blob` module — `BlobStorage` (`BTreeMap<Vec<u8>, BlobValue>` behind `Arc<RwLock<…>>`), `BlobValue` (`Raw` / `Compressed` / `Tombstone`), `compress_if_worthwhile` helper, `should_compress` predicate, and `COMPRESS_THRESHOLD_BYTES = 200`.
- `AliceDB::put_blob(&[u8], &[u8]) -> io::Result<()>` — insert or overwrite a byte-keyed blob; ALICE-Zip zlib compresses payloads that are at least 200 bytes long *and* actually shrink meaningfully (16-byte savings margin).
- `AliceDB::get_blob(&[u8]) -> io::Result<Option<Vec<u8>>>` — exact-key lookup; tombstones and missing keys both surface as `None`.
- `AliceDB::scan_blob_prefix(&[u8]) -> io::Result<Vec<(Vec<u8>, Vec<u8>)>>` — byte-lex prefix walk skipping tombstones.
- `AliceDB::delete_blob(&[u8])` — LSM tombstone semantics; the entry becomes immediately invisible.
- `AliceDB::blob_len()` / `AliceDB::blob_is_empty()` — live-key statistics.
- 4 unit tests in `src/blob.rs::tests` covering compression threshold and worthiness heuristics.
- 9 integration tests in `tests/blob_kv.rs` covering round-trip on short and long payloads, missing keys, deletion, prefix ordering, empty-prefix scan, overwrite, concurrent writers (8 threads × 64 keys), and coexistence with the time-series engine.

### Design decisions

- **Threshold parity with ALICE-CodeTracker v0.4.0**: the 200-byte cutoff matches the tracker's jsonl backend so payloads that cross the ALICE-Zip integration are compressed identically on either side.
- **Independent MemTable**: blob storage lives in its own `RwLock`-guarded `BTreeMap` alongside the existing time-series `MemTable`. Neither system reads the other's state.
- **Alpha-1 is in-memory only**: no WAL, no SSTable format v2, no compaction — those arrive in alpha-2 and alpha-3. Callers that need durability today must not rely on the blob store surviving process restarts. See `docs/EXPANSION_PROPOSAL.md` §5 for the roadmap.

### Downstream

The primary motivator is [ALICE-CodeTracker v0.5.0+](https://github.com/ext-sakamoro/ALICE-CodeTracker), which will add an `alice-tracker-alicedb` backend against this API once the durability story is in place (alpha-2 milestone).

## [0.1.0] - 2026-02-23

### Added
- `model` — `ModelType` (Polynomial, Fourier, SineWave, MultiSine, PerlinNoise, Constant, Linear, RawLzma), `DataType`, `FitResult`
- `memtable` — `MemTable` with BTreeMap buffer and automatic flush-to-segment via `FitConfig`
- `segment` — `DataSegment` model-based SSTable with rkyv zero-copy serialization
- `storage_engine` — `StorageEngine` LSM-tree architecture with WAL, compaction, and mmap segment reads
- `query_engine` — `QueryBuilder`, `Aggregation` (Sum/Avg/Min/Max/Count/First/Last/StdDev/Variance), `QueryResult`
- `analytics_bridge` — (feature `analytics`) `AnalyticsSink` for ALICE-Analytics metric persistence
- `crypto_bridge` — (feature `crypto`) encryption at rest via ALICE-Crypto
- `sdf_bridge` — (feature `sdf`) SDF spatial data with Morton code indexing
- Python bindings (feature `python`) — PyO3 + NumPy zero-copy, `AliceDB` class with context manager
- `AliceDB` high-level API: `open`, `put`, `put_batch`, `get`, `scan`, `aggregate`, `downsample`
- 86 unit tests
