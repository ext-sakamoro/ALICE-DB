# G13 HashMap/HashSet audit manifest

G13 = no unspecified iteration order: any VALUE (not just internal
bookkeeping) that a caller can observe must not depend on
`std::collections::HashMap`/`HashSet` iteration order, since that order is
randomized per-instance within a process and is not guaranteed stable
across runs. Fix, where real, is `BTreeMap`/`BTreeSet` (ordered by key) --
not pinning the hasher.

This file covers `src/replication.rs`, `src/storage_engine.rs`,
`src/lib.rs`, and `src/fs_audit.rs`. `src/transaction.rs` is covered in its
own section below (merged from a concurrently-run change on a separate
branch): its table uses its own column layout and a Japanese-language
write-up, with labels that correspond to this file's `stays` / `fixed`
vocabulary as `(a) membership-only / 反復順不問` = `stays` and
`(b) 反復順が戻り値を左右する` = `fixed` (no `out of scope` sites in that
file). Kept as originally written rather than translated, to not
introduce transcription errors into already-reviewed content.

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

---

# G13 HashMap/HashSet 反復順序監査 (`src/transaction.rs`)

決定論 gate G13 (unspecified iteration order 禁止) のための `src/transaction.rs`
全 22 サイトの分類記録。

`std::collections::HashMap`/`HashSet` の反復順序は型として保証されない。`RandomState`
がプロセス内でもインスタンスごとに乱択されるため、同じ論理内容の `HashMap` を
複数回構築すると、構築のたびに `.keys()`/`.values()`/`.iter()` の反復順序が変わり
うる (実測: 同一挿入順で fresh 構築した `HashMap<u64, u64>` 20 個中 20 個が互いに
異なる `.keys()` 順序を示した)。本 crate の目的は bit 一致の再現可能な結果なので、
戻り値・シリアライズされる値・ログ・副作用の順序が反復順に依存する箇所は全て
`BTreeMap`/`BTreeSet` (キー順序で反復順が一意に定まる) に置き換える。

分類基準:

- **(a) membership-only / 反復順不問**: 呼び出しが全て `.get()` / `.get_mut()` /
  `.insert()` / `.remove()` / `.contains_key()` / `.contains()` / `.entry()` の
  point 操作のみ、または反復されていても (1) 各要素を独立に変更するだけで
  共有アキュムレータが無い、(2) 唯一観測可能な結果が反復順に依存しない `usize`
  (件数) である、のいずれか。`HashMap`/`HashSet` のまま。
- **(b) 反復順が戻り値を左右する**: 反復した結果 (戻り値・ログ・副作用の順序) が
  反復順に依存する。`BTreeMap`/`BTreeSet` に変換。

## (a) membership-only (変更なし)

| field | 型 | 宣言行 | 判定根拠 (全 call site を読んで確認) |
|---|---|---|---|
| `MvccStore::versions` | `HashMap<String, Vec<MvccEntry<T>>>` | L38 | `read`/`write`/`commit`/`abort` は全て `get`/`get_mut`/`entry` による point アクセス (L95, L128, L145, L169, L196)。`lock_count_internal` は `.len()` (件数、反復順不問、L215)。`gc()` のみ `.values_mut()` で全走査する (L224) が、各 `chain` (`Vec`) を独立に `drain`/`retain` するだけで、戻り値 `removed` は `before_len - chain.len()` の総和 (`usize` の加算は可換、どの順で足しても同じ値)。他の key の処理結果に影響しない。 |
| `MvccStore::txn_status` | `HashMap<TxnId, TxnStatus>` | L40 | 全 call site が `.get()` (L108, L123, L132, L160, L189, L209) または `.insert()` (L84, L179, L202)。`gc()` 内の `chain.retain(|e| self.txn_status.get(&e.txn_id) != ...)` (L243) も `retain` のクロージャから呼ばれる point lookup で、`txn_status` 自体を反復していない。 |
| `MvccStore::txn_start_ts` | `HashMap<TxnId, u64>` | L42 | `.insert()` (L85) と `.get()` (L94) のみ。反復なし。 |
| `MvccStore::txn_write_set` (外側) | `HashMap<TxnId, HashSet<String>>` | L44 | 外側の `HashMap` は `.insert()` (L86) / `.get_mut()` (L147) / `.get(..).cloned()` (L167, L194) のみ。反復なし。 |
| `MvccStore::txn_write_set` (内側の `HashSet<String>`、`write_set` 変数) | — | L44 の value 型 | `commit()` (L167-176) と `abort()` (L194-199) が `for key in &write_set` で反復する。各 `key` は `self.versions` の異なる key を指すので、`key` ごとの処理 (`entry.commit_ts = commit_ts` への代入、または `chain.retain`) は他の `key` の処理と独立。共有アキュムレータなし、戻り値もなし (`()`)。反復順不問。 |
| `LockManager::locks` | `HashMap<String, LockEntry>` | L285 | `.entry()` (L312-319、`acquire`) / `.get_mut()` (L362、`release`) は point。`detect_deadlock()` (L405) と `lock_count()` (L465-468) のみ全走査するが、前者は `wait_for` への insert (集合メンバシップの構築、挿入順不問で可換) のみに使い、後者は `.filter().count()` (件数、反復順不問)。 |
| `LockManager::held_locks` (外側) | `HashMap<TxnId, HashSet<String>>` | L287 | `.entry()` (L325-328, L349-352, L372-375) / `.get_mut()` (L364) / `.remove()` (L395) は point。`release_all()` (L386-389) のみ `ks.iter().cloned().collect()` で内側 `HashSet<String>` を `Vec<String>` へ複製するが、後述のとおりこの順序も結果に影響しない。 |
| `LockManager::held_locks` (内側の `HashSet<String>`、`ks`/`held` 変数) | — | L287 の value 型 | `release_all()` は複製した `Vec<String>` の各 `key` を順に `self.release(txn_id, key)` へ渡す。`release()` は `self.locks.get_mut(key)` で**その `key` の `LockEntry` のみ**を変更し (`waiters` の昇格も同じ `key` の `LockEntry` 内で完結)、他の `key` の `LockEntry` や `held_locks` の他エントリに触れない。異なる `key` 間に共有状態が無いため、`release()` を呼ぶ順序を変えても最終状態と戻り値 `count` (`keys.len()`、L391) は変わらない。 |
| `LockManager::LockEntry::holders` | `HashSet<TxnId>` | L274 | `.is_empty()` (L321, L368) / `.insert()` (L323, L348, L370) / `.contains()` (L332) / `.len()` (L336) / `.remove()` (L363) は point/件数。`detect_deadlock()` の `for &holder in &entry.holders` (L407) は `wait_for` への insert (集合メンバシップ構築、可換) にのみ使われ、反復順は `wait_for` の内容に影響しない。`lock_count()` の `.filter(|e| !e.holders.is_empty())` (L467) も `is_empty()` の point 呼び出し。 |

## (b) 反復順が戻り値を左右する (`BTreeMap`/`BTreeSet` に変換)

| field / 変数 | 旧型 | 新型 | 宣言行 | 判定根拠 |
|---|---|---|---|---|
| `detect_deadlock::wait_for` | `HashMap<TxnId, HashSet<TxnId>>` | `BTreeMap<TxnId, BTreeSet<TxnId>>` | L403→L412 | `wait_for.keys()` (L418→L427) が DFS の開始ノードを決める外側ループの反復対象。複数の循環が独立に存在しうるグラフでは、どのノードから DFS を始めるかで**どの循環が最初に見つかり、`Some(..)` として返るか**が変わる。 |
| `detect_deadlock::visited` | `HashSet<TxnId>` | `BTreeSet<TxnId>` | L414→L423 | `wait_for`/`graph` と同じ DFS 呼び出しに渡される同種の txn id 集合。それ自体は `.insert()`/`.contains()` の point 操作のみで反復されないが、`dfs_cycle` のシグネチャ・型の対称性のため `wait_for`/`graph` と合わせて変換する。 |
| `detect_deadlock::in_stack` | `HashSet<TxnId>` | `BTreeSet<TxnId>` | L415→L424 | 同上 (`visited` と同じ理由)。 |
| `dfs_cycle::graph` (引数) | `&HashMap<TxnId, HashSet<TxnId>>` | `&BTreeMap<TxnId, BTreeSet<TxnId>>` | L442 | `wait_for` を受け取る引数。`graph.get(&node)` で得た `neighbors` を `for &next in neighbors` で反復する順序が、DFS がどのノードを先に訪問するか・`path` に何がどの順で積まれるかを決め、循環検出時に返す `path[start..].to_vec()` (戻り値そのもの) の内容・回転順を左右する。 |
| `dfs_cycle::visited` (引数) | `&mut HashSet<TxnId>` | `&mut BTreeSet<TxnId>` | L444 | 呼び出し元の `visited` と同じ型でなければならない。 |
| `dfs_cycle::in_stack` (引数) | `&mut HashSet<TxnId>` | `&mut BTreeSet<TxnId>` | L445 | 呼び出し元の `in_stack` と同じ型でなければならない。 |

`visited`/`in_stack` は単体では `.insert()`/`.contains()`/`.remove()` の point 操作しか
受けず、それ自体が反復されることはない (正直な注記: 技術的には `HashSet` のまま
でも `visited`/`in_stack` 単体の非決定性は生じない)。ただし `wait_for`/`graph` が
運ぶ非決定性と不可分の DFS 呼び出しの一部であり、同じ `TxnId` ドメインの集合として
型を揃えておくことが将来の変更 (例えば `visited` を反復する機能追加) に対して
安全なため、本修正では `wait_for`/`graph` と合わせて `BTreeSet` に統一した。

## 件数突合

grep `HashMap\|HashSet` (import 行含む) = 22 行。内訳:
import 1 行 + (a) 7 field の宣言+初期化 (各 2 行 = 14 行) + `holders` の宣言+初期化
(2 行、`held_locks`/`locks` と重複カウントしない) + (b) 4 箇所 (宣言 2 + 関数シグネチャ
3 箇所のうち 1 つは同一行に `HashMap`/`HashSet` 両方を含む)。詳細は本ファイル冒頭の
表のとおり、全 22 grep 行を (a) 7 field + (b) 4 サイト (型注釈としては `wait_for`
宣言 1 + `visited`/`in_stack` 宣言 2 + `dfs_cycle` シグネチャ中の `graph`/`visited`/
`in_stack` 3 = 延べ 6 箇所、修正後の行番号では L412, L423, L424, L442, L444, L445
の 6 行) に割り当て済み。未分類のまま残った行は無い。
