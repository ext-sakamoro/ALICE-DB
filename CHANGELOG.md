# Changelog

All notable changes to ALICE-DB will be documented in this file.

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
