# Changelog

All notable changes to ALICE-DB will be documented in this file.

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
