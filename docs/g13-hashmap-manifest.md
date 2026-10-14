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
