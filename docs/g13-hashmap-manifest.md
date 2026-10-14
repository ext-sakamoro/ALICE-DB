# G13 HashMap/HashSet audit manifest

G13 = no unspecified iteration order: any VALUE (not just internal
bookkeeping) that a caller can observe must not depend on
`std::collections::HashMap`/`HashSet` iteration order, since that order is
randomized per-instance within a process and is not guaranteed stable
across runs. Fix, where real, is `BTreeMap`/`BTreeSet` (ordered by key) --
not pinning the hasher.

This file covers `src/replication.rs`, `src/storage_engine.rs`,
`src/lib.rs`, and `src/fs_audit.rs`. `src/transaction.rs` is covered by a
separate, concurrently-run change on its own branch; merge that file's
findings into this table when the two branches are reconciled -- no
attempt was made here to coordinate column layout or scope with it
beyond using the same classification vocabulary (`stays` / `fixed` /
`out of scope`).

## Legend

- **stays**: HashMap/HashSet is correct here; no caller-observable value
  depends on iteration order (membership-only bookkeeping, or every
  access is a point `get`/`insert`/`remove`, never a direct `.iter()` /
  `for` over the map/set itself).
- **fixed**: was order-dependent; changed to `BTreeMap`/`BTreeSet` in this
  pass.
- **out of scope**: not production code (test-only tooling gated out of
  normal builds); G13 does not apply.

## `src/replication.rs` -- 5 sites, all **stays**

`RaftNode` fields:

```rust
pub next_index: HashMap<NodeId, LogIndex>,
pub match_index: HashMap<NodeId, LogIndex>,
```

Both are leader-only Raft follower-progress tracking (membership
bookkeeping, not a computed/returned value). Every call site in the file
was read (not just grepped) and confirmed to be one of:

| # | Site | Operation |
|---|------|-----------|
| 1 | `RaftNode::new` | `HashMap::new()` (construction) |
| 2 | `become_leader` | `.insert(peer, next)` / `.insert(peer, LogIndex(0))` for `peer` drawn from `self.peers: Vec<NodeId>` (already deterministic insertion order) -- never iterates the maps themselves |
| 3 | `build_append_entries` | `self.next_index.get(&peer)?` (point get) |
| 4 | `handle_append_response` | `.insert(peer, resp.match_index)` / `.insert(peer, LogIndex(...))` / `.get_mut(&peer)` (point ops) |
| 5 | `try_advance_commit` | `self.match_index.get(&peer).is_some_and(...)` inside `for &peer in &self.peers` (point get per peer from the `Vec`; the `HashMap` itself is never iterated) |

`clone_for_test()` (test-only helper) does `.clone()` of both maps, which
does not read or depend on iteration order either.

**No `.iter()`, `.keys()`, `.values()`, or `for x in &self.next_index` /
`for x in &self.match_index` exists anywhere in this file.** Grep used to
re-verify coverage: `grep -n "match_index\.\|next_index\." src/replication.rs`
(12 occurrences, all accounted for above) plus a full read of the file
(772 lines) to confirm no indirect iteration (e.g. via a helper that
takes `&HashMap` and iterates it) exists either.

Conclusion: stays `HashMap`, no change.

## `src/storage_engine.rs` + `src/lib.rs` -- 1 site, **fixed** (same field, 2 declarations)

`StorageEngine::stats()` builds a per-model-type count by folding over
`self.shared.index` (confirmed `BTreeMap<u64, SegmentIndexEntry>`, so the
fold's accumulation order was already deterministic) into a
`HashMap<String, usize>`. The fold's *process* was deterministic; the
*result type* was not order-stable when later inspected by a caller
(`{:?}`, equality against a snapshot, serialization, etc.), because two
`HashMap` instances holding the same keys get independently randomized
per-instance hasher state.

Changed:

- `src/storage_engine.rs`: local `model_counts` fold target, and the
  `pub struct StorageStats { ..., model_distribution: ... }` field --
  both `HashMap<String, usize>` -> `BTreeMap<String, usize>`.
- `src/lib.rs`: `pub struct PyStats { ..., model_distribution: ... }`
  (the pyo3 binding wrapper for `StorageStats`, `#[pyo3(get)]` field) --
  same change. The conversion site (`fn stats(&self) -> PyResult<PyStats>`
  in the `python` module, which does
  `model_distribution: stats.model_distribution`) needed no further
  change since both sides are now the same type.

pyo3 BTreeMap support: this crate pins `pyo3 = "0.29"`, resolved to
`0.29.2` in `Cargo.lock`. Verified in the vendored source
(`~/.cargo/registry/src/*/pyo3-0.29.2/src/conversions/std/map.rs`) that
`IntoPyObject`/`FromPyObject` are implemented for
`collections::BTreeMap<K, V>` with the same bounds as the `HashMap`
impls in the same file -- no workaround was needed, this is not a
blocker.

Oracle test added: `storage_engine::tests::test_stats_model_distribution_order_independent`
(see test output / mutation result in the final report). Builds two
`StorageEngine`s that flush the same two data patterns (a constant
series and a linear ramp) in two different orders ending on the same
model-type multiset, and asserts both the value (`==`) and the `{:?}`
string are identical between them.

Conclusion: fixed, `HashMap` -> `BTreeMap` in both declarations.

## `src/fs_audit.rs` -- 5 sites, **out of scope**

Module doc (line 1): "Open-file audit for the file backend (test builds
only)." Gated entirely behind `#[cfg(feature = "open-file-audit")]`; the
feature is `[]` (no default enablement) and is only turned on via this
crate's own dev-dependency on itself for its test suite
(`Cargo.toml`: `alice-db = { path = ".", default-features = false,
features = ["open-file-audit"] }` under `[dev-dependencies]`). Module doc
explicitly states "not meant to be enabled by dependants."

The 5 sites:

| # | Site | Type | Operation |
|---|------|------|-----------|
| 1 | `open_handles()` | `HashMap<PathBuf, usize>` behind `OnceLock<Mutex<...>>` | construction |
| 2 | `ARMED` thread_local | `HashSet<String>` | construction |
| 3 | `note_open` | `HashMap` | `.entry(...).or_insert(0)` (point) |
| 4 | `note_close` / `is_open` | `HashMap` | `.get_mut(&k)` / `.remove(&k)` / `.contains_key(&k)` (point) |
| 5 | `arm_crash_point` / `disarm_all_crash_points` / `take_crash_point` | `HashSet` | `.insert(name)` / `.clear()` / `.remove(name)` (point, no iteration) |

No `.iter()`/`.keys()`/`.values()` exists anywhere in this module either
(confirmed by reading the full 209-line file). Even setting that aside:
this module tracks open file handles and armed crash-injection points
purely so this crate's *own* test suite can catch Windows file-handle
bugs during compaction/rename/truncate. There is no path from this
module's `HashMap`/`HashSet` state into any computed, stored, or returned
*production* value -- it is test infrastructure, not a library behavior
observable by a non-test caller of `alice-db`. This is a stronger
exclusion than "membership-only production code" (the replication.rs
verdict above): it is not production code at all, analogous to how G10's
wall-clock audit excluded `#[cfg(test)]` code.

Conclusion: out of scope for G13, no change. (Documented here per the
instruction to record this as a considered-and-excluded conclusion
rather than a silent omission.)

## Pending merge

- `src/transaction.rs` findings (separate, concurrently-run change) are
  not included in this file. When the two branches are reconciled,
  append that file's table(s) here using the same `stays` / `fixed` /
  `out of scope` vocabulary.
