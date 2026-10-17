# G13: unspecified order (HashMap / HashSet, threads)

G13 = no value a caller can observe may depend on an unspecified order: the iteration
order of `std::collections::HashMap` / `HashSet` (randomised per instance), or the order
in which threads finish.

## Collections

The crate holds no `HashMap` / `HashSet`. The earlier sites, all membership-only or
order-independent bookkeeping (the Raft follower progress in `src/replication.rs`, the MVCC
version chains, transaction status and lock tables in `src/transaction.rs`, the open-file
audit in `src/fs_audit.rs`), use `BTreeMap` / `BTreeSet`, so any iteration added later runs
in key order. `StorageStats::model_distribution` and the deadlock detector's wait-for graph
were already `BTreeMap` (each with an order-independence test).

`clippy.toml` bans `HashMap`, `HashSet` and their iterator types, so a new site does not
compile; the clippy canary plants every entry.

## Threads and parallel work

`scripts/order_guard.py` reads `src/` (production code) and `Cargo.toml` for threads, rayon
and crossbeam, and fails on any use not listed in `scripts/order_allowlist.toml` with the
exact line count, the SHA-256 of each line and the reason:

| Site | Why the order cannot reach an output |
|------|--------------------------------------|
| `src/memtable.rs` `rayon::join` | two independent fits of the same values, compared afterwards in a fixed order |
| `src/storage_engine.rs` background flush thread | one thread takes flush jobs from a bounded FIFO channel in send order |
| `Cargo.toml` `rayon`, `crossbeam-channel` | the two sites above |

`clippy.toml` bans `std::thread::spawn`, `scope`, `Builder::spawn`, `spawn_scoped`,
`available_parallelism` and the `Builder`, `JoinHandle`, `ScopedJoinHandle` types; the flush
thread and the concurrency tests carry an `allow` with their reason at the statement.
