# Changelog

All notable changes to ALICE-DB will be documented in this file.

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
