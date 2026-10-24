# ALICE-DB

[English](README.md)

数値時系列のためのモデルベース LSM-tree データベース memtable を flush する時に、
バッファした点へ [ALICE-Zip](https://github.com/ext-sakamoro/ALICE-Zip) で候補モデル
(多項式、Fourier、sine、Perlin noise、定数、一次式、LZMA による fallback) を当てはめ、
サンプルの代わりにそれを記述するモデルを保存する 点・範囲の問い合わせはモデルを評価して
答える 時系列とは別に、値を任意の `i64` key でビットのまま保存する exact series (モデルを
使わない)、byte をキーにした blob ストア、当てはめた法則を証拠・成立範囲・後から来た証拠の
判定と一緒に保存する law ストアを持つ

保存先はディレクトリ (WAL、セグメントのファイル、advisory lock、既定で有効な `fs` feature)
かプロセスのメモリ (`AliceDB::in_memory`、`wasm32-unknown-unknown` でもビルドできる)
のどちらかで、API は両者で同じ

License: LicenseRef-Proprietary

## 向かない用途

- **既定での厳密な保存** 既定の `FitConfig` は非可逆で、誤差がモデルの種類ごとに文書化した
  閾値を下回れば採用する (例えば単一 sine は相対 MSE < 0.1) 厳密なビットで読み戻すには
  `FitConfig` の `lossless: true` で XOR 残差を保存する 残差は `原信号 ^ モデル` なので、
  読む側が同じ算術で法則を評価した時だけ原信号に戻る したがって blob はその算術の識別子を
  持つ 別の算術を名乗る残差に当たった読み出しは値を返さず、`ResidualError::Mismatch` (segment、
  その key 範囲、store、両方の識別子) を持つ `InvalidData` の error で失敗する 残差を適用すると
  原信号でも model でもない値になり、model だけでは原信号でないため 点ごとの補正に復号できない
  残差も `ResidualError::Corrupt` として同じく断る 範囲読みは全件か error のどちらかで、件数の
  少ない結果や一部だけ補正した結果は返さない 数えるのはまだ key に答える segment だけで、全 key が
  より新しい書き込みで上書きされた古い segment は範囲読みを失敗させない
  `StorageConfig::residual_policy = ResidualPolicy::ModelOnly` (segment では
  `query_point_model_only` / `query_range_model_only`) で model の近似を明示して読める
  `residual_status()` は読まずにどの場合かを報告する **別の算術で書いた store の厳密な値を
  取り戻すにはその算術が要る** alice-db 0.3 (alice-det-math 0.4) が書いた store は
  alice-db 0.3 で厳密に読めるので、そこで読んで本版で書き直す 本版だけでは model の値しか得られない
  **壊れた残差** (`ResidualError::Corrupt`) はどの版でも取り戻せない 補正が失われているので、
  segment を backup から戻す (または `ModelOnly` で model を受け入れる)
  0.3.0 より前に書かれた残差は識別子を持たず、適用される 書いた機械では厳密だがそれを他の場所で
  確かめる手段が無い
  任意の key で厳密に読み戻す必要がある値は [exact series](#exact-series) に置く
- **モデル保存での不規則なタイムスタンプ** セグメントは点ごとのタイムスタンプを持たず、
  最初と最後のタイムスタンプの間の等間隔の格子で読み戻す 既定 (非可逆) では欠けのある系列は
  点ごとには再現しない lossless では間隔の変わる所で新しいセグメントを切るので、増加する
  タイムスタンプは間隔によらず厳密に読み戻る (小さいセグメントに分かれ圧縮は効かない) 既に
  書いたタイムスタンプ以下を書くと、不正確に保存する代わりに `LosslessKeyOrderError` で拒む
- **一般的な SQL やドキュメント用途** 値は `i64` タイムスタンプごとの `f32` と、不透明な
  blob と、法則だけ

## 目次

- [インストール](#インストール)
- [使用例](#使用例)
- [Exact series](#exact-series)
- [法則: 証拠・成立範囲・判定](#法則-証拠成立範囲判定)
- [保存先](#保存先)
- [Feature](#feature)
- [モデルの種類](#モデルの種類)
- [Python](#python)
- [最小サポート Rust バージョン](#最小サポート-rust-バージョン)
- [ビルドとテスト](#ビルドとテスト)
- [関連 crate](#関連-crate)
- [ライセンス](#ライセンス)

## インストール

```sh
cargo add alice-db
# ブラウザ / ファイルシステムなし: メモリの保存先のみ
cargo add alice-db --no-default-features
```

## 使用例

```rust
use alice_db::{Aggregation, AliceDB, StorageConfig};

// `AliceDB::open("./my_data")` keeps the same data in files (`fs` feature)
let db = AliceDB::in_memory(StorageConfig::default())?;

// y = 0.5 t + 10 at t = 0..1000
for t in 0..1000_i64 {
    db.put(t, 0.5 * t as f32 + 10.0)?;
}
db.flush()?; // fit a model to the buffered points and store it as a segment

// point query, computed from the stored model
let v = db.get(500)?.expect("t = 500 was stored");
assert!((v - 260.0).abs() < 1e-2);

// range aggregation
let avg = db.aggregate(0, 999, Aggregation::Avg)?;
assert!((avg - 259.75).abs() < 1e-2);
# Ok::<(), std::io::Error>(())
```

## Exact series

厳密に読み戻す必要がある記録 (メトリクス、シミュレーションの step、疎な key や順不同の key) は
モデル保存ではなく名前付きの series に置く 1 点が blob 保存の 1 record になり、`f32` / `f64` の
値はビットを保ち、scan は書いた key だけを key 順で返し、名前が違えば互いの点は見えない

```rust
use alice_db::{AliceDB, StorageConfig};

let db = AliceDB::in_memory(StorageConfig::default())?;
let energy = db.series("energy")?;
for (i, step) in [0i64, 1, 1000, 1_000_003, 1 << 40, (1 << 53) - 1].into_iter().enumerate() {
    energy.put_f32(step, 0.1 + i as f32 * 1.7)?;
}
assert_eq!(energy.get_f32(1000)?, Some(0.1 + 2.0 * 1.7));
assert_eq!(energy.scan_f32(0, 1 << 41)?.len(), 5);
# Ok::<(), std::io::Error>(())
```

`analytics` feature はメトリクスを series `analytics_bridge::METRICS_SERIES` に書く

## 法則: 証拠・成立範囲・判定

`alice_db::law_store` は ALICE-Zip の `SignalLaw` (module 内で再公開) を保存する
`SignalLaw` は多項式の `y = f(x)` に、当てはめに使った点、その点で測った残差、点が覆う `x`
の範囲、出典、参照値を持たせたもの

| メソッド | 動作 |
|----------|------|
| `put_law(name, &law, &semantics_id)` | `name` の次の版として保存し、内容の識別子で index に入れる |
| `get_law(name)` / `get_law_version(name, v)` | 版を復元する 残差は保存した証拠から測り直す 結果は `RecordSemantics` を持つ: `Current` / `VerifiedPrevious(id)` / 形式 2 より前の記録は `Unpinned` この build が再現しない識別子の記録は拒否する (`LawRecordMismatch`、両方の識別子を示す) `get_law_with(name, LawReadPolicy::RequirePinned)` は `Unpinned` も拒否する |
| `evaluate_law(name, x)` | 最新版の `f(x)` (同じ `RecordSemantics` 付き) 成立範囲外の `x` は拒否し (`LawError::OutOfRange`)、外挿しない |
| `ingest_evidence(name, points, &policy, &semantics_id)` | 新しい点を判定し (証拠なし / 範囲外 / 支持 / parameter 更新 / 残差増加 / 破綻)、判定を記録する parameter 更新は新しい版として保存し、古い版も読める |
| `law_pointers_by_id(&id)` / `evaluate_by_id(&id, x)` | 1 つの内容識別子の下にある `(name, version)` を全部返す、またそこを通して `f(x)` を返す |
| `law_history(name)` / `law_versions(name)` / `law_names()` | 記録した判定 (証拠の点数と RMS 付き) を順に、保存した版、保存した名前 |

```rust
use alice_db::law_store::{IngestPolicy, Provenance, SignalLaw, Verdict};
use alice_db::{AliceDB, SEMANTICS_ID, StorageConfig};

// `SEMANTICS_ID` は法則を評価する算術の識別子 点評価器を提供する crate から
// re-export しているので、この build が実際に使う算術の識別子そのものになる
// (自前の定数で置き換えない) 法則の内容識別子に入るので、保存した結果は
// 「どの法則か」と「どの算術で計算したか」の両方を名乗る

let db = AliceDB::in_memory(StorageConfig::default())?;
// y = 1 + 2x measured at x = 0..=4
let pts: Vec<(f64, f64)> = (0..5).map(|i| (f64::from(i), 1.0 + 2.0 * f64::from(i))).collect();
let law = SignalLaw::fit_polynomial(&pts, 1, Provenance::new("run 1", "least squares"))?;
db.put_law("line", &law, &SEMANTICS_ID)?;

assert!((db.evaluate_law("line", 2.5)?.value - 6.0).abs() < 1e-12);
assert!(db.evaluate_law("line", 9.0).is_err()); // outside [0, 4]

// The stored law is reachable by its content identifier as well as by name.
let id = law.law_id(&SEMANTICS_ID);
assert_eq!(db.law_pointers_by_id(&id)?, vec![("line".to_string(), 1_u64)]);

let policy = IngestPolicy { abs_tolerance: 0.01, break_factor: 4.0 };
let v = db.ingest_evidence("line", &[(0.5, 2.0), (3.5, 8.0)], &policy, &SEMANTICS_ID)?;
assert!(matches!(v, Verdict::Supports { .. }));
assert_eq!(db.law_history("line")?.len(), 1);
# Ok::<(), Box<dyn std::error::Error>>(())
```

法則は blob ストアの予約キー接頭辞 `"\0alice-law\0"` の下にレコードとして置くので、ファイルの
保存先では blob WAL / `SSTable` に書かれ、`to_bytes` にも含まれる レコードはチェックサムを持ち、
自分のキーを記録している 壊れたレコード、別のキーへ複製されたレコード、`SignalLaw::from_parts`
が拒否するレコード (例えば保存した範囲の外にある証拠) はエラーとして返す バイト配置は
`law_store` module に記載している `cargo run --example law_store` で両方の保存先について
一連の流れを実行できる


### 法則を内容で引く

名前は人が付けるもので、後から変わりうる `SignalLaw::law_id` (ALICE-Zip) は
法則そのもの (成立範囲と係数) と、評価に使う数値意味論の識別子から計算される
32 byte の識別子なので、保存した結果が「どの法則から出たか」を名前に依存せず
名指しできる

`put_law` と `ingest_evidence` はその識別子を受け取り、書き込む版をすべて
index に入れる `law_pointers_by_id` は 1 つの識別子に対する `(name, version)` を
全部返し、`evaluate_by_id` はそれを評価する 複数の pointer があっても一意に
定まるのは、識別子が同じなら評価が全ての `x` で同じ bit を返すため 逆向きは
成立しない (評価が同じでも識別子は分かれうる) ので、識別子は重複排除の鍵には
使えない

## 保存先

| 保存先 | コンストラクタ | 備考 |
|--------|----------------|------|
| ファイル (`fs`) | `AliceDB::open(path)`、`AliceDB::with_config(config)` | セグメントファイル + index、時系列 WAL、blob WAL と `SSTable`、`SSTable` の mmap 読み出し、blob WAL の排他 advisory lock |
| メモリ | `AliceDB::in_memory(config)`、`AliceDB::from_bytes(config, &bytes)` | ファイルシステム不要 `to_bytes` はデータベース全体 (どちらの保存先でも) をチェックサム付きの 1 つのバッファとして書き出す |

同じ書き込み列はどちらの保存先からもビット単位で同じに読み戻せる
(`tests/storage_backend_parity.rs`)

セグメントのファイルは開く時に 1 度だけ、書いた側と同じセグメントの型に decode し、メモリ上のセグメントと同じ code で問い合わせる (`tests/segment_view_parity.rs`) そのため開いたセグメントはファイルとほぼ同じ大きさのメモリを使う (model の係数と residual の bytes) ファイルを map したままにはしない セグメントのファイルは保存形式の番号を持つ (形式 1: header の後に rkyv 0.8 の archive) 0.6.0 より前が書いたファイル (形式 0: header の無い rkyv 0.7 の archive) も読む

ファイルの保存先は、開く時に blob の WAL の末尾の切れた record や壊れた record を、最後の正しい record の位置で切る (その後の書き込みが壊れた部分の後ろに隠れて失われないように) 落とす bytes は切る前に WAL の隣の `blob.wal.dropped-<offset>` に書いて fsync し、`AliceDB::blob_wal_recovery()` が何を切ったかを返す 同じ位置での後の回復は前の写しを置き換えず (`blob.wal.dropped-<offset>.<n>` に書く)、書き終えた写しは消さない 記録の境界で終わる WAL は触らない

回復は WAL を最初の壊れた record の手前までだけ残し、それより後ろは使わない そのため log の途中の record が壊れると、その後に書いて ack した record も live の store から落ちる それらは落とした写しにだけ残る (長さを信用できない record の先は replay しない)

## Feature

| Feature | 既定 | 説明 |
|---------|------|------|
| `fs` | yes | ファイル保存 (`open` / `with_config`: WAL、mmap、advisory lock) 無効にするとメモリの保存先だけになる |
| `ffi` | no | C / C++ / C# FFI (`fs` を含む)、ヘッダは `bindings/` |
| `python` | no | Python バインディング (PyO3 + NumPy、`fs` を含む) |
| `analytics` | no | ALICE-Analytics 連携: 集計したメトリクスを exact series へ (`fs` を含む) |
| `crypto` | no | ALICE-Crypto による保存時暗号化 (`crypto_bridge::EncryptedDB`、`fs` を含む) |
| `sdf` | no | Morton code で索引する SDF 空間データの保存 (`fs` を含む) |

`analytics` と `crypto` は crates.io の公開版 `alice-analytics` 0.3 と `alice-crypto` 0.4 に依存する

## モデルの種類

| モデル | 向く形 |
|--------|--------|
| `Constant` | 平坦な区間 |
| `Linear` | 傾斜 |
| `Polynomial` | 曲線、ドリフト |
| `Fourier` | 周期信号 |
| `SineWave` / `MultiSine` | 振動 |
| `PerlinNoise` | ノイズ状の模様 |
| `RawLzma` | どのモデルも採用されない時の fallback |

モデルは当てはまりの良さとサイズでセグメントごとに選ぶ 圧縮率と問い合わせ速度はデータに
依存するので、`cargo bench` で対象のハードウェアで測る

## Python

```sh
pip install maturin
maturin develop --release --features python
```

```python
import alice_db

db = alice_db.open("./my_timeseries")
timestamps = list(range(1000))
values = [i * 0.5 for i in range(1000)]
db.put_batch(timestamps, values)  # 長さの等しい 2 列
print(db.get(100), db.aggregate(0, 999, "avg"))
db.close()
```

`open` の既定は lossy なモデル保存 (`lossless=False`) タイムスタンプに gap のある系列は
等間隔 grid で読み戻されるため、flush 後に gap の点の値が変わることがある `lossless=True` を
渡すと残差を保存し、増加するタイムスタンプを厳密に読み戻す (Rust の `FitConfig { lossless: true }`
と同じ約束):

```python
db = alice_db.open("./exact", lossless=True)
db.put_batch([0, 1, 2, 10], [10.0, 20.0, 30.0, 31.0])
db.flush()
print([db.get(t) for t in (0, 1, 2, 10)])  # [10.0, 20.0, 30.0, 31.0] 書いた全点が厳密
db.close()  # lossy 既定は gap の内側がずれる (例: get(1) は 20.0 でなく 10.0)
```

## 最小サポート Rust バージョン

Rust 1.87 (`Cargo.toml` の `rust-version`) CI でライブラリを既定 feature、native の feature
一式、既定 feature なしで確認している 1.86 は 1.87 を宣言する依存 crate が拒否する 開発と CI
は `rust-toolchain.toml` で固定した toolchain を使う

## ビルドとテスト

```sh
cargo test                                   # file backend (default)
cargo test --no-default-features             # memory backend only
cargo test --features ffi,sdf                # native feature set
cargo run --example law_store
cargo clippy --all-targets -- -W clippy::pedantic -D warnings
cargo bench
scripts/preflight.sh            # every CI step with the same arguments
scripts/preflight.sh --quick    # static checks, clippy, wasm build, docs and `cargo test --lib`
```

## 関連 crate

| Crate | 関係 |
|-------|------|
| [ALICE-Zip](https://github.com/ext-sakamoro/ALICE-Zip) | モデルの当てはめと生成器、`law::SignalLaw` |
| [ALICE-DetMath](https://github.com/ext-sakamoro/ALICE-DetMath) | どの target でも同じ bit を返す超越関数 (bloom filter の大きさ、ALICE-Zip 経由の法則評価)、`SEMANTICS_ID` |
| [ALICE-Analytics](https://github.com/ext-sakamoro/ALICE-Analytics) | ストリーミング集計 (`analytics` feature) |
| [ALICE-Crypto](https://github.com/ext-sakamoro/ALICE-Crypto) | 保存時暗号化 (`crypto` feature) |
| [ALICE-Edge](https://github.com/ext-sakamoro/ALICE-Edge) | 組込み機器でのモデル当てはめ |

<!-- law-integration:begin law=68dc0c3c26591798325f73da79762b95d2d663ae -->
## ALICE-Law との統合

ALICE-Law の `scripts/gen-law-readme-block.py` が ALICE-Law `68dc0c3c2659` とこの repository の `alice-law-conformance.toml` から生成する 手で編集しない

| | |
|---|---|
| release train | alice-db 0.5.1 (published)、Law への依存: あり |
| 算術 | train の alice-det-math 0.5.1、算術の識別子 `85e0884623b154a3b2b264374907b6cbac09307ca3f9a08339e48c567d3cb315` この crate の要求は `alice-det-math = "0.5.1"` |
| end to end 検査 | 対象 (`alice-db = "0.5"`)、digest `9f7cf676708a272e3817ebbf0727c610f391e1547747af4307ee44d5523267c2`、aarch64 と x86_64 で同じ |

### 適合 (G1-G14)

| gate | 状態 | 注記 |
|---|---|---|
| G1 no-libm-direct | 適合 | every banned method of the G1 list is in clippy.toml, every feature is linted with -D warnings (python included, without pedantic: the pyo3 binding trips … |
| G2 ieee-max-min | 適合 | raw f32 / f64 max, min and clamp are a default-deny in clippy.toml, with no exception and so no raw_* manifest; the helpers come from alice-det-math, the owner … |
| G3 no-outer-fneg-of-fma | 適合 | the library's only fma calls are the fit competition's model score (not a stored value); the 114 fma calls in the IR otherwise come from alice-det-math and … |
| G4 debug-release-arch-goldens | 適合 | boundary: a stored value read back (get / scan from the memtable, or from a segment through its residual: lossless, or a lossy-configured segment holding a … |
| G5 release-profile-test-leg | 適合 | the aarch64 release leg is macos-latest. The repository has no branch protection; a leg is required in the sense that main moves only by a fast-forward after … |
| G6 semantics-id | 適合 | alice-db computes its own identifier, DB_SEMANTICS_ID (e1a82c64...37c4ce with alice-zip 0.11), from the bits of the decisions it computes itself and the … |
| G7 handshake-carries-id | 適合 | the Law-bearing value that crosses this crate's boundary is the stored law record (alice-zip SignalLaw parts), handed over in AliceDB::to_bytes / from_bytes … |
| G8 persisted-semantics-id | 適合 | every durable form that carries a value computed under an arithmetic carries that arithmetic's identifier: lossless segment residuals since 0.3.0 (the … |
| G9 persisted-meaning-change-is-major | 適合 | the audit covers the identifier this crate persists and finds no violation, and tests/identifier_history.rs reads a committed store stamped with each earlier … |
| G10 no-run-varying-artifacts | 適合 | no ambient clock or OS-entropy read is known in a Law-bearing path: `python3 scripts/entropy_guard.py` reports `24 files, 10 patterns, 0 problems` … |
| G11 gate-self-tests | 未達 | unmet: clause (c). scripts/clippy_canary.py, scripts/test_clippy_canary.py, scripts/gen_clippy_canary_calls.py, scripts/test_gen_clippy_canary_calls.py and … |
| G12 public-repo-text-rules | 適合 | the exemptions in .public-text.toml are counted per file and class (the documentation linter's own word list and its tests) |
| G13 no-unspecified-order | 適合 | the order-dependent outputs found earlier (model_distribution, detect_deadlock) were fixed with an oracle each; the remaining HashMap / HashSet sites, … |
| G14 float-environment-invariants | 適合 | the static scan covers the files of this repository and the dynamic check covers the builds CI runs on ubuntu and macOS; a flag passed by a build outside this … |

### feature と保証の範囲

| feature | 保証 | 注記 |
|---|---|---|
| `alice-analytics` | 未申告 |  |
| `analytics` | 未申告 |  |
| `crypto` | 未申告 |  |
| `ffi` | 未申告 |  |
| `fs` | 未申告 |  |
| `numpy` | 未申告 |  |
| `open-file-audit` | 未申告 |  |
| `pyo3` | 未申告 |  |
| `python` | 未申告 |  |
| `sdf` | 未申告 |  |
<!-- law-integration:end -->

## ライセンス

この版から `LicenseRef-Proprietary`: ソースは閲覧用に公開しており、利用・複製・改変・配布には
著作権者との書面での合意が必要 条文は [LICENSE-Proprietary](LICENSE-Proprietary)
それより前に公開した版は、公開時のライセンスのまま

ライセンスの問い合わせ: <contact@extoria.co.jp>
