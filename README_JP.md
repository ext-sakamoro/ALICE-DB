# ALICE-DB

[English](README.md)

数値時系列のためのモデルベース LSM-tree データベース memtable を flush する時に、
バッファした点へ [ALICE-Zip](https://github.com/ext-sakamoro/ALICE-Zip) で候補モデル
(多項式、Fourier、sine、Perlin noise、定数、一次式、LZMA による fallback) を当てはめ、
サンプルの代わりにそれを記述するモデルを保存する 点・範囲の問い合わせはモデルを評価して
答える 時系列とは別に、byte をキーにした blob ストアと、当てはめた法則を証拠・成立範囲・
後から来た証拠の判定と一緒に保存する law ストアを持つ

保存先はディレクトリ (WAL、mmap したセグメント、advisory lock、既定で有効な `fs` feature)
かプロセスのメモリ (`AliceDB::in_memory`、`wasm32-unknown-unknown` でもビルドできる)
のどちらかで、API は両者で同じ

License: AGPL-3.0-or-later OR LicenseRef-Commercial

## 向かない用途

- **既定での厳密な保存** 既定の `FitConfig` は非可逆で、誤差がモデルの種類ごとに文書化した
  閾値を下回れば採用する (例えば単一 sine は相対 MSE < 0.1) 厳密なビットで読み戻すには
  `FitConfig` の `lossless: true` で XOR 残差を保存する 残差は `原信号 ^ モデル` なので、
  読む側が同じ算術で法則を評価した時だけ原信号に戻る したがって blob はその算術の識別子を
  持ち、別の算術を名乗る残差は適用せずに飛ばす (`segment::residual_semantics` が報告する)
  0.3.0 より前に書かれた残差は識別子を持たず、書いた機械では厳密だがそれを他の場所で
  確かめる手段が無い
- **不規則なタイムスタンプ** セグメントは最初と最後のタイムスタンプの間を等間隔と仮定するので、
  欠けのある系列は点ごとには再現しない
- **一般的な SQL やドキュメント用途** 値は `i64` タイムスタンプごとの `f32` と、不透明な
  blob と、法則だけ

## 目次

- [インストール](#インストール)
- [使用例](#使用例)
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

## 法則: 証拠・成立範囲・判定

`alice_db::law_store` は ALICE-Zip の `SignalLaw` (module 内で再公開) を保存する
`SignalLaw` は多項式の `y = f(x)` に、当てはめに使った点、その点で測った残差、点が覆う `x`
の範囲、出典、参照値を持たせたもの

| メソッド | 動作 |
|----------|------|
| `put_law(name, &law, &semantics_id)` | `name` の次の版として保存し、内容の識別子で index に入れる |
| `get_law(name)` / `get_law_version(name, v)` | 版を復元する 残差は保存した証拠から測り直す |
| `evaluate_law(name, x)` | 最新版の `f(x)` 成立範囲外の `x` は拒否し (`LawError::OutOfRange`)、外挿しない |
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

assert!((db.evaluate_law("line", 2.5)? - 6.0).abs() < 1e-12);
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
| ファイル (`fs`) | `AliceDB::open(path)`、`AliceDB::with_config(config)` | セグメントファイル + index、時系列 WAL、blob WAL と `SSTable`、mmap 読み出し、blob WAL の排他 advisory lock |
| メモリ | `AliceDB::in_memory(config)`、`AliceDB::from_bytes(config, &bytes)` | ファイルシステム不要 `to_bytes` はデータベース全体 (どちらの保存先でも) をチェックサム付きの 1 つのバッファとして書き出す |

同じ書き込み列はどちらの保存先からもビット単位で同じに読み戻せる
(`tests/storage_backend_parity.rs`)

## Feature

| Feature | 既定 | 説明 |
|---------|------|------|
| `fs` | yes | ファイル保存 (`open` / `with_config`: WAL、mmap、advisory lock) 無効にするとメモリの保存先だけになる |
| `ffi` | no | C / C++ / C# FFI (`fs` を含む)、ヘッダは `bindings/` |
| `python` | no | Python バインディング (PyO3 + NumPy、`fs` を含む) |
| `analytics` | no | ALICE-Analytics 連携: 集計したメトリクスを時系列へ (`fs` を含む) |
| `crypto` | no | ALICE-Crypto による保存時暗号化 (`crypto_bridge::EncryptedDB`、`fs` を含む) |
| `sdf` | no | Morton code で索引する SDF 空間データの保存 (`fs` を含む) |

`analytics` と `crypto` は隣のリポジトリ `../ALICE-Analytics` と `../ALICE-Crypto` に依存する

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
db.put_batch([(i, i * 0.5) for i in range(1000)])
print(db.get(100), db.aggregate(0, 999, "avg"))
db.close()
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

## ライセンス

`AGPL-3.0-or-later OR LicenseRef-Commercial` のデュアルライセンスで、どちらかを選ぶ

| 選択肢 | 条件 | 使う場面 |
|--------|------|----------|
| **AGPL-3.0-or-later** | [LICENSE-AGPL](LICENSE-AGPL)、無償 | 自分のプロジェクトも AGPL 互換のオープンソースである、または内部でのみ使う |
| **Commercial License** | [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md)、有償、コピーレフトを外す | クローズドソース製品、プロプライエタリな SaaS、ファームウェアやプラグインの配布 |

AGPL は強いコピーレフトで、`alice-db` をリンクして配布する、または利用者にサービスとして
提供する製品・ファームウェア・サービスは AGPL で公開する必要がある

商用ライセンスの問い合わせ: <contact@extoria.co.jp>
