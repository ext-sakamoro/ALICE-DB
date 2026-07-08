# ALICE-DB Expansion Proposal — General LSM-KV Path Alongside Time-Series

**Status**: Design proposal (2026-07-08)
**Author**: Moroya Sakamoto (`sakamoro@alicelaw.net`)
**Motivation**: ALICE-CodeTracker v0.4.0 needed a general blob store backend but ALICE-DB v0.1.0 only exposes `(i64 timestamp, f32 value)` time-series semantics. Rather than reach for `sled` / `redb` / `rocksdb`, this proposal sketches how ALICE-DB itself can grow a general LSM-KV path while preserving its time-series marquee feature.

## 1. Current scope (v0.1.0)

ALICE-DB is a **Model-Based LSM-Tree Database** for numerical time-series data:

- **Public API**: `put(timestamp: i64, value: f32)` / `get(timestamp: i64) -> Option<f32>` / `scan(start, end) -> Vec<(i64, f32)>` / `aggregate` / `downsample`
- **Core insight**: Kolmogorov complexity-inspired compression — store the mathematical model (`y = 0.5x + 10`) that generates the data, not the raw points
- **Compression ratio**: 50–1000× for structured numerical data (sensor readings, financial ticks, physics simulation output)
- **Architecture**: MemTable (`BTreeMap<i64, f32>`) → Fitter Competition (Polynomial / Fourier / Perlin / …) → Segment SSTable (model + residual)
- **License**: AGPL-3.0-only (private repo, `publish = false`)

## 2. The gap ALICE-CodeTracker v0.4.0 surfaced

The tracker stores `(Uuid, Stub)` records where `Stub` is:

```rust
pub struct Stub {
    pub id: Uuid,                                    // 16 bytes
    pub fingerprint: StubFingerprint,                // ast_path + 32-byte blake3 + kind
    pub file: String,                                // ~30–200 bytes
    pub line: u32,
    pub column: Option<u32>,
    pub kind: StubKind,                              // enum
    pub language: Language,                          // enum
    pub snippet: String,                             // ~30–2000 bytes (heaviest field)
    pub crate_name: Option<String>,
    pub commit_hash: Option<String>,
    pub llm_session: Option<LlmSession>,             // ~100 bytes when Some
    pub created: DateTime<Utc>,
    pub resolved_at: Option<DateTime<Utc>>,
    // …
    pub load_bearing: Vec<String>,
    pub relocation_history: Vec<Relocation>,          // ~50 bytes/entry
}
```

None of these fields fit into `f32`. The tracker's fields fall into three groups:

1. **Numeric time-series** (would benefit from ALICE-DB's current path): `created` / `resolved_at` counts over time, aggregated cost per LLM per day, compressible if binned.
2. **Rich blobs** (needs a KV path): `snippet` / `ast_path` / `file` / `relocation_history` — pure text and small JSON arrays.
3. **Indexes** (needs range/point lookup on strings): fingerprint keys, file prefix scans, crate name filters.

Time-series ALICE-DB handles (1) natively. It cannot express (2) or (3) without a general blob store. The initial v0.4.0 plan bundled a new `alice-tracker-alicedb` backend on this assumption and had to be dropped.

## 3. Proposed expansion — Path X (LSM-KV alongside time-series)

Add a **blob store column family** to ALICE-DB that reuses the existing LSM infrastructure but bypasses the model fitter for keys/values that are not compressible via polynomial or Fourier models.

### 3.1 New public API

```rust
impl AliceDB {
    /// Existing time-series API (unchanged).
    pub fn put(&self, timestamp: i64, value: f32) -> io::Result<()> { … }
    pub fn get(&self, timestamp: i64) -> io::Result<Option<f32>> { … }
    pub fn scan(&self, start: i64, end: i64) -> io::Result<Vec<(i64, f32)>> { … }

    // === v0.2.0 additions ===

    /// Blob key-value put. `key` is treated as bytes; equality and prefix-scan are
    /// the only relations. `value` is compressed via ALICE-Zip zlib on the write
    /// path (skipped when it fails to beat 1.05×).
    pub fn put_blob(&self, key: &[u8], value: &[u8]) -> io::Result<()>;

    /// Blob get by exact key.
    pub fn get_blob(&self, key: &[u8]) -> io::Result<Option<Vec<u8>>>;

    /// Prefix scan for blob keys. Ordered by lexicographic key.
    pub fn scan_blob_prefix(&self, prefix: &[u8]) -> io::Result<Vec<(Vec<u8>, Vec<u8>)>>;

    /// Delete a blob by key. LSM tombstone (physical removal on next compaction).
    pub fn delete_blob(&self, key: &[u8]) -> io::Result<()>;
}
```

### 3.2 Storage layer changes

The LSM-tree already has `MemTable` (BTreeMap-backed), WAL, and SSTable format. Extending them:

| Layer          | Time-series (existing)              | Blob KV (new)                                     |
|----------------|-------------------------------------|---------------------------------------------------|
| MemTable       | `BTreeMap<i64, f32>`                | `BTreeMap<Vec<u8>, BlobValue>` in parallel        |
| WAL            | `TS { ts, val }` records            | `Blob { key, value_or_tombstone }` records        |
| SSTable format | model + residuals + index           | key-sorted blocks + ALICE-Zip-compressed values   |
| Fitter         | Poly / Fourier / Perlin / …         | Bypassed (blob path never fits models)            |
| Compaction     | Merge sorted by timestamp           | Merge sorted by key (bytes-lex)                   |

`BlobValue` is a tagged union:

```rust
enum BlobValue {
    Raw(Vec<u8>),          // < compression-threshold or high-entropy
    Compressed(Vec<u8>),   // zlib payload, decompressed on read
    Tombstone,             // deletion marker (LSM semantics)
}
```

### 3.3 Where ALICE-Zip fits

The blob path is where ALICE-Zip becomes the compression engine (`alice_core::compression::zlib_compress` for now, later `lzma_compress` for large fields). The tracker's `snippet` (30–2000 bytes) compresses 2–4× on real workloads. Same call-site as ALICE-CodeTracker v0.4.0's `alice-tracker-jsonl::compression`.

Threshold heuristic (mirrored from tracker v0.4.0): only compress when `value.len() >= 200`. Below that, zlib overhead (~11 bytes headers) makes raw store cheaper.

### 3.4 What ALICE-DB does NOT lose

The marquee feature (Kolmogorov model compression for time-series) is untouched. The blob path is opt-in via distinct method names (`put_blob` vs `put`) and lives in a distinct SSTable family. Users who only care about IoT sensor streams see zero API change and pay no runtime cost for blob support (unused code paths are cold).

## 4. Downstream consumers this unlocks

Beyond ALICE-CodeTracker, the same blob path serves several ALICE-* crates that currently reach for `sled` / `redb` / raw SQLite:

| Crate                | Current backend                | Would benefit from ALICE-DB blob path?          |
|----------------------|--------------------------------|--------------------------------------------------|
| `ALICE-CodeTracker`  | JSONL / SQLite / D1 / Turso    | ✅ Primary motivator                             |
| `ALICE-Metrics`      | in-process HashMap             | ✅ Time-series **+** blob metadata is a natural fit |
| `ALICE-Audit`        | SQLite                         | ✅ Blob audit logs with LSM-friendly append shape |
| `ALICE-Log`          | JSONL / rotating file          | ✅ Blob logs compressed by ALICE-Zip natively    |
| `ALICE-Font`         | rkyv static file               | ⚠️ Read-only, LSM overkill                       |
| `ALICE-Bamboo`       | 3MF / STL files                | ⚠️ Binary blobs but write-once, not LSM-shaped   |

The three ✅ crates (`ALICE-Audit` / `ALICE-Log` / `ALICE-Metrics`) all currently maintain their own storage layer; consolidating on ALICE-DB is a real dogfooding win.

## 5. Migration path from v0.1.0

- **v0.1.0 → v0.2.0-alpha** (~30–40 h): add blob API surface, MemTable split, WAL variant, SSTable format v2, blob-family compaction. Time-series API and file format unchanged; existing DBs continue to work.
- **v0.2.0 → v0.2.1** (~10 h): add secondary index (prefix trie or hash-lookup Bloom) for `scan_blob_prefix` acceleration.
- **v0.3.0** (~20 h): promote blob path to first-class marquee feature in README, add explicit `alice-db --blob-first` mode for consumers who never need time-series.

## 6. Risk register

- **API bloat**: doubling the surface area (`put` / `get` / `scan` × 2) is a documentation and learning-curve hit. Mitigation: publish blob API in a separate `alice_db::blob` module so time-series consumers do not see it in autocomplete.
- **SSTable format churn**: format v2 is a breaking change for existing on-disk DBs. Mitigation: keep v1 as a supported read-only mode, provide `alice-db migrate` tool.
- **Compaction complexity**: two column families = two mergers; care needed to avoid interleaving that starves either family. Mitigation: per-family compaction thread pool with a shared budget.
- **Dogfooding lock-in**: if ALICE-CodeTracker depends on ALICE-DB, a bug in ALICE-DB stops the tracker from monitoring itself and every other ALICE-* crate. Mitigation: keep `jsonl` backend as the always-available fallback, treat the ALICE-DB backend as opt-in via `[storage] backend = "alicedb"` in `alice-tracker.toml`.

## 7. Effort estimate

| Milestone                                    | Hours | Cumulative |
|----------------------------------------------|-------|------------|
| Design finalisation + RFC review              | 5     | 5          |
| MemTable / WAL / SSTable format v2 groundwork | 15    | 20         |
| Blob API + compaction (single-family shim)    | 12    | 32         |
| Fuzz + property tests + concurrency stress    | 8     | 40         |
| ALICE-CodeTracker `alice-tracker-alicedb` backend | 6 | 46         |
| ALICE-Metrics / Audit / Log adaptation      | 12    | 58         |
| Docs + release notes + migration tool         | 4     | 62         |

Ballpark: **~40 h for ALICE-DB v0.2.0-alpha**, **~60 h to fully retire ALICE-CodeTracker's other backends** in favour of ALICE-DB.

## 8. Decision point

This proposal is **not scheduled**. It documents the path if and when we decide to consolidate ALICE-* persistence on ALICE-DB. In the meantime:

- **ALICE-CodeTracker v0.4.0** shipped with `alice-tracker-jsonl` compression using ALICE-Zip. The dogfooding link is established at the compression layer already.
- **ALICE-DB v0.1.0** remains focused on time-series; consumers with `(i64, f32)` semantics (`ALICE-Physics` sensor sweeps, `ALICE-TRT` benchmark results) are the current priority.
- The next natural forcing function is when a second consumer needs the blob path. Two consumers is enough signal to schedule v0.2.0-alpha.

## 9. Alternatives considered

- **Add blob support to a new sibling crate `ALICE-KV`**: rejected because it fragments the persistence story across two crates. If we ever unify, we would just move `ALICE-KV` into `ALICE-DB` anyway.
- **Have ALICE-CodeTracker embed `redb` / `sled`**: rejected as the "cheap short-term" option because it locks the tracker (and any future ALICE-* consumer) out of the ALICE-Zip integration and forgoes the dogfooding synergy.
- **Byte-encode Stub records into `(timestamp, f32)` chunks**: rejected as pathological — recording a Stub JSON as a hundred `f32` fake-samples defeats the compression model and is not queryable.

## 10. Cross-references

- `~/ALICE-CodeTracker/CHANGELOG.md#040` — v0.4.0 release notes documenting the deferral of the ALICE-DB backend
- `~/ALICE-CodeTracker/docs/audits/STUB_AUDIT_v0.4.0.md` — audit trail confirming this deferral
- `~/ALICE-CodeTracker/crates/shared/alice-tracker-jsonl/src/compression.rs` — the ALICE-Zip integration that establishes the dogfooding link
- `~/ALICE-Zip/libalice/src/compression/mod.rs` — the compression API this proposal reuses
