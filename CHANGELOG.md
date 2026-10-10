# Changelog

All notable changes to ALICE-DB will be documented in this file.

## [Unreleased]

## [0.3.0] - 2026-10-11

### Fixed

- 時系列 store の範囲読み (`AliceDB::scan` / `query_range`、`DataSegment` と zero-copy の `SegmentView`、多項式の SIMD 経路を含む) が終わらないことがあった f64 の時刻を格子の step ずつ足しており、2^53 を超える key で step が ulp の半分より小さいと時刻が進まず、結果を伸ばし続けて メモリを使い切った (2^62 付近の連続した key で再現) 格子点を整数の index で歩き、時刻は `query_start + ⌊k·range/(n−1)⌋` を 128 bit 整数で正確に出す 1 回の読みが segment ごとに返す点は `point_count` 以下 値を求める sample 位置は従来と同じ f64 演算になる形で出すので、整数の step で 2^53 未満の key では読み戻す bit は変わらない (`tests/determinism_golden.rs` の digest は不変)
- segment の時刻の差 (`end − start`、`t − start`) を i64 で引いていた 幅が `i64::MAX` を超える segment (正負をまたぐ key) で debug build は panic、release build は折り返して別の位置を読んでいた i128 で引く 退化した segment (`point_count ≤ 1` で幅が正) は従来すべての整数時刻を返していたが、先頭の 1 点だけを返す
- `AliceDB::get` が memtable を読まず、flush 前に書いた値が `None` になっていた 複数の segment が key を覆う時は index が最初に返した segment が答えていた memtable (同じ key は最後の書き込み) を先に読み、次に segment を新しい順 (id の大きい順) に読む `scan` も memtable の点を含め、同じ key は memtable、次に新しい segment の値を残す
- `FitConfig { lossless: true }` で間隔の揃わない key を書くと、別の key の値が返っていた segment は key を持たず両端の間の等間隔の格子で読み戻すので、残差が書いていない key に当たる 疎な step (0, 1, 1000, 1000003, 2^40, 2^53−1) では 2^40 までの全点が先頭の値 0.1 を返し、範囲 0..=2^41 が 5 行中 1 行だった 判断: 1 segment のまま正確にするには segment に点ごとの時刻を持たせる保存形式の変更が要る (別途) ので、lossless mode の memtable は間隔の変わる所で segment を切る 増加する key は間隔によらず bit 単位で読み戻る (gap の多い系列は小さい segment に分かれ圧縮は効かない) 既に書いた key 以下の key は不正確に保存する代わりに `LosslessKeyOrderError` で拒む (下の Changed) lossless の segment は格子点だけを持つものとして読み、格子点でない key の `get` は `None`、`scan` はその segment の最初の格子点から歩く 既定の非可逆 mode の挙動は変えない
- analytics ブリッジが書いた metric を読み戻すと別の key の値が返っていた metric key は metric の間に最大 2^44 の間隔を空けるので、上の格子の仮定に合わない (140 metric × 6 variant の 840 key で一致 0 件、flush 後は key の幅の引き算が debug で panic) 書き込み先を exact series に移した (下の Changed)
- `AnalyticsSink::persist` が何も flush していなかった metric は書いた時点で読め、`persist` は書いた後に blob WAL を sync するので、blob WAL の sync 方針 (`Batched` / `Manual` を含む) によらず返った時点で永続 `flush_db` は blob WAL と時系列の両方を flush する
- `analytics_bridge` の module doc が key の式を `% 0xFFFFF` と書いていた 実装どおり `& 0xFFFFF` に直した
- `scripts/docs_lint.py` が空の `[Unreleased]` (release を切った直後の状態) を「比較 0 件」として失敗にしていた そのため release のたびに `[Unreleased]` へ中身の無い行を置く必要があった 空の `[Unreleased]` は、`Cargo.toml` の版にその版の節がある時だけ通す 版を上げたのに節が無く `[Unreleased]` も空の時は失敗にし、本文があるのに category の見出しが無い時は従来どおり失敗にする
- Fuzz の workflow が crash を見つけても成功していた (run の step が `continue-on-error`) crash で job を失敗させ、各 target が 1 件以上の入力を実行したことを確かめ (0 件は失敗)、target ごとの実行数と coverage を job summary に出す crash の入力は失敗時に artifact として残す
- Fuzz の 3 target (`fuzz_query_parse` / `fuzz_serialize_roundtrip` / `fuzz_index_lookup`) は crate を呼ばない scaffold で、実行数は出ても何も試していなかった 公開 API を呼ぶ形にした: `from_bytes` に任意の byte 列と `to_bytes` → `from_bytes` の往復 / blob の put / get / delete / prefix scan を `BTreeMap` と突合 / query builder に任意の範囲・集計・group by・limit・offset (手元 90 秒で coverage 3939 / 1895 / 2159、crash 無し) 各 target に coverage の下限を置き (`fuzz/coverage-floor.txt`、`scripts/fuzz_reach.py`、下限の無い target と空の下限表は失敗)、crate を呼ばない target や checksum で止まる target が通らないようにした `fuzz/regressions/<target>` の入力を毎回再生する
- `fuzz_index_lookup` を 1 byte = 1 操作の program にした (上位 3 bit が操作、下位 3 bit が 1〜2 byte の固定の key か生の key、put の値は位置ごとに異なる) 構造化した入力では「同じ key を書いて flush、もう一度書いて flush」に届かず、注入した 4 種の欠陥 (get が古い SSTable を先に読む / scan の優先順位の反転 / compaction で古い値が勝つ / 特定の key で get が Err) を空の corpus から 60 秒で 1 つも見つけられなかった 書き換え後は手元 (負荷の高い機械) の 3 回ずつで 3/3 / 3/3 / 3/3 / 2/3 flush・compaction・開き直しの後は触った全 key を get で読み、全体を scan する その列を表す入力 (2 回の flush をまたぐ同じ key、削除して flush した key の復活) と見つかった crash を `fuzz/regressions/fuzz_index_lookup/` に置く 決定的な試験 `tests/blob_merge_semantics.rs` (6 列: 2 つの SSTable で新しい値が勝つ / 削除の flush 後に復活しない / memtable が 3 つの SSTable を隠す / prefix scan で key ごとに解決 / flush 前の削除が compaction と開き直しを越えて残る / prefix と等しい key がその prefix scan に入る) を足した (fuzz は代わりにならない) CI の fuzz の一時 directory を tmpfs (`/dev/shm`) にする
- `fuzz_index_lookup` は操作が `Err` を返すと黙って飛ばしていた (`if let Ok`) 正しい key への操作は全て成功を要求し、`Err` も欠陥として落とす 一時 directory の store で、memtable を新しい SSTable に書き出す・全 SSTable を merge する・開き直すを入力に混ぜ、複数 SSTable の merge と新しい値 (と削除) が勝つ扱いを `BTreeMap` と突合する (手元 180 秒で coverage 1895 → 3204、不一致無し)
- `fuzz_law_record` が record 末尾の CRC32 を越えられず、本体の parser に届いていなかった (180 秒で coverage 33) 入力をそのままに加え、magic・format・入力・正しい CRC32 で包んだ形でも decode する (coverage 482、2380 万件で crash 無し)
- blob `SSTable` (`blob_sstable.rs`) と blob WAL (`blob_wal.rs`) の形式の記述が実装と合っていなかった checksum は CRC-32C と書いていたが、実装 (`crc32fast`) は CRC-32 (IEEE 802.3、多項式 `0xEDB88320`) を計算している SSTable の header の版は「currently 1」と書いていたが、書き手は 3 を書き、読み手は 1 / 2 / 3 を読む 記述には bloom section、24 byte の footer (`records_size` / `bloom_size` / magic)、tombstone の `value_kind = 0x02` も無かった 形式も実装も変えず、記述を実装に合わせた (既存の file はそのまま読める) `tests/blob_format_pin.rs` は 1 件の record を書いた SSTable と WAL の byte 列を、独立に実装した CRC-32 (IEEE) と照合し、module doc が checksum と版を実装どおりに書いているかも検査する 確認: record / bloom / WAL の CRC、版、reserved の値、doc の版の行の 6 つの変異でどれも red
- `crypto_bridge` の `test_encrypted_wal_replay` が常に失敗していた 書き込み後の「クラッシュ」を `AliceDB` の `mem::forget` で模していたため blob WAL の advisory lock が同じ process に残り、開き直しが `WouldBlock` になる (v0.2.0-alpha.3 で lock を入れて以降) CI は `crypto` feature を build していなかったので表に出なかった 暗号化 WAL を持つ storage engine の層でクラッシュを模す形に直した
- README / README_JP / `examples/law_store.rs` の使い方が、法則の内容識別子に渡す値として架空の定数 (`const SEMANTICS: [u8; 32] = [0x11; 32]`) を使っていた crate が `alice_db::SEMANTICS_ID` として 実物 (この build が実際に使う算術の識別子) を re-export しているので、3 箇所すべてそれに差し替えた 架空の値を使うと、保存した法則が「どの算術で計算したか」を名乗らず、別の build で復元した時に同じ識別子を再現できない
- README / README_JP の ```rust block が何の検査も通っていなかった `src/lib.rs` の `#[cfg(doctest)]` で両 README を doctest として取り込み、`ci.yml` と `scripts/preflight.sh` に `cargo test --doc` の step を足した (公開 doc の構成は変えない) 実測: 両 README の 2 block ずつ計 4 本が doctest として実行され、`put_law` を存在しない method 名に替えると `no method named ... found` で FAILED になる

### Added

- 値を bit 単位で保存する exact series: `AliceDB::series(name)` が返す `Series` に `put_f32` / `put_f64` / `get` / `get_f32` / `get_f64` / `scan` / `scan_f32` / `scan_f64` / `delete` 1 点が blob key-value store の 1 record で、model の fit はしない 値は f32 / f64 の bit のまま (NaN の payload と符号付き 0 も)、`scan(start, end)` は書いた key だけを key 順で返す key は `i64` (この crate の `put` / `get` / `scan` と、step 番号・metric key・epoch 時刻の呼び出し側に合わせた) 保存形式は `"\0alice-series\0" 名前 "\0"` + 符号 bit を反転した big-endian の key 8 byte (byte 順 = 符号付きの順)、値は 4 byte (f32) か 8 byte (f64) の little-endian で、幅で型を区別し、別の幅で読むと `InvalidData` 名前ごとに名前空間が分かれる (空の名前と NUL を含む名前は `InvalidInput`) 両方の保存先で使え、`to_bytes` / `from_bytes` にも載る
- `LosslessKeyOrderError` (下の Changed)、`SegmentView::is_lossless` / `grid_ceil` / `on_grid`、`MemTable::get` / `range` / `breaks_spacing`
- analytics ブリッジの読み出し: `read_metric` / `scan_metric` / `metric_name_hash` / `MetricPoint` / `METRICS_SERIES`、`AnalyticsSink::read` / `scan`
- 試験: `tests/exact_series.rs` (疎な key・負の key・i64 の両端、f32 / f64、開き直しと compaction と snapshot、名前空間、key の配置) / `tests/analytics_metrics_exact.rs` (840 値の bit 一致、2 回の flush、file の開き直し、メモリ保存、record の配置) / `tests/segment_range_arith.rs` (2^62 付近と正負をまたぐ key と i64 全域の範囲読み、3 秒の期限を越えたら process を失敗で終える) / `tests/get_reads_memtable.rs` / `tests/lossless_irregular_keys.rs` / `tests/sparse_steps_repro.rs` (physics 側の再現と同じ key と値を plain の lossless と exact series で、メモリ保存と file + 開き直し) いずれも修正前の main で red を確認し、各修正を戻す変異 (f64 の step / i64 の引き算 / get が memtable を飛ばす / 重なる segment の古い方を読む / ブリッジが時系列 store に書く / key の byte 順の反転 / lossless の切り分けを外す / 順序の検査を外す) でそれぞれ red になる
- `tests/crash_reopen_process.rs`: `AliceDB` を開いた子 process が WAL 有効で 500 点を書き、全 `put` が返ったと報告した後に kill される (Unix は `SIGKILL`、Windows は `TerminateProcess`、`close` も `Drop` も走らない) 親が同じ設定で開き直し、全点を bit 単位で読み戻す (暗号化 WAL も `crypto` feature で同様) 同じ process 内の試験は `mem::forget` でクラッシュを模すため advisory lock が残り、`AliceDB` の層の再オープンを通せなかった 本 file は試験の追加で、製品の挙動は変えていない WAL への書き込みを止める変異で 2 本とも red を確認
- `alice_db::SEMANTICS_ID`: the identifier of the arithmetic models are
  evaluated with (re-exported from `alice-zip`, equal to
  `alice_det_math::SEMANTICS_ID`), so a caller writing a residual with
  `segment::compress_residual_xor` does not need a direct `alice-zip`
  dependency to name it
- `tests/determinism_golden.rs`: SHA-256 over the exact IEEE 754 encodings of
  model selection and the fitted parameters (lossy and lossless, six fixed
  series), `query_point` / `query_range` output, every aggregate over the
  values read back, bloom filter sizing
  (`num_bits` / `num_hashes` over ten false positive rates and 1 to 5,000,000
  elements), `law_id` with the evaluation it stands for, and the
  `SEMANTICS_ID` hex. Each scenario fails when it hashed fewer bytes than its
  floor, and the file runs in every feature set of the CI test job on every OS

### Changed

- **破壊的変更 (`analytics` feature):** `alice-analytics` を 0.2 から 0.3 に上げた `analytics_bridge` の `flush_metrics_to_db` / `AnalyticsSink` が受け渡す `MetricPipeline` などは依存先の版ごとに別の型なので、`alice-analytics` 0.2 の型を作って渡している利用者は compile できなくなる (利用者も 0.3 に上げる) 0.3 は `alice-det-math` 0.4 を要求するので、依存グラフの det-math が 1 版 (0.4.0) になり、保存した Law の評価と分析側の推定値が同じ算術で計算される (従来は 0.3.2 と 0.4.0 が同居していた) `alice-analytics` は path + version の依存から公開版への依存に替え、CI は analytics ブリッジを実物に対して build する
- **破壊的変更 (`analytics` feature):** `flush_metrics_to_db` / `AnalyticsSink` は metric を時系列 store (`AliceDB::put_batch`) に書かず、exact series `"alice-analytics/metrics"` (`METRICS_SERIES`) の `metric_key` に書く 関数の signature は変えていないが、metric を `AliceDB::get` / `scan` で読んでいた利用者は何も読めなくなる `read_metric` / `scan_metric` / `AnalyticsSink::read` / `scan` で読む ブリッジに渡す値 (counter / gauge / cardinality / p50 / p90 / p99 の f32) は従来どおりで、0.3.0-beta.2 と同じ event 列に対し `flush_metrics_to_db` に渡る 720 値が全て一致することは確認済 (HLL と DDSketch の `ln64` / `exp64` も det-math 0.3.2 と 0.4.0 で 400 万入力ずつ一致) **0.3.0-beta.x が時系列 store に書いた metric は移行しない** 時系列 store に残り、`AliceDB::get` / `scan` で `metric_key` を引けば読めるが、上の Fixed の理由で別の key の値が混ざる (正確な値は取り戻せない)
- **破壊的変更 (lossless mode):** `FitConfig { lossless: true }` の `put` / `put_batch` は、既に書いた key 以下の key を `io::ErrorKind::InvalidInput` の `io::Error` (中身は `LosslessKeyOrderError { key, last }`) で拒む batch は全体を先に検査し、拒んだ呼び出しは何も書かない 開き直し後は保存済みの最大 key から続く 従来は受け付けて別の key の値を返していた 上書きや順不同の key が要る値は exact series に置く 既定の非可逆 mode は従来どおりどの順の key も受け付ける
- 依存を `alice-zip` 0.7 → 0.8、`alice-crypto` 0.1 → 0.4 に上げた `alice-crypto` は path + version (`../ALICE-Crypto`, `"0.1"`) から crates.io の公開版への依存に替えた (隣の checkout は既に 0.3.0 で、要求 `"0.1"` を満たさず build できなかった) 本 crate が使う `seal` / `open` / `derive_key` / `hash` / `Key` / `CipherError` に変更は無い (0.3.0 と 0.4.0 の公開版で `stream.rs` / `hash.rs` / `kdf.rs` と依存は同一、0.4.1 は crate-type から `cdylib` を外しただけなので、暗号化した WAL と record の byte 列は変わらない) 公開済の 0.3.0-beta.2 は `alice-zip` ^0.7 を要求していたため、他の ALICE crate と組むと `alice-zip` が 0.7 と 0.8 の 2 版 link されていた 保存済みの記録に関わる値は変わらない: `tests/determinism_golden.rs` の `SEMANTICS_ID`・model 選択・`law_id` の digest は 0.8 でも同じ値で通る
- CI と `scripts/preflight.sh` の `NATIVE_FEATURES` に `analytics` と `crypto` を加えた それまで CI は `alice-crypto` を空の stub crate に置き換えており、`analytics` も対象外だったので、どちらの bridge も CI で compile されていなかった stub を作る `.github/actions/alice-stubs` は削除、`cargo hack` の除外は `python` だけにした `analytics_bridge` の clippy pedantic の指摘 (`# Errors` 節 5 件、test 3 件) を直した
- `alice-det-math` 0.4 is a direct dependency (the version `alice-zip` 0.8
  and `alice-analytics` 0.3 use). Bloom filter sizing takes its logarithm from `alice_det_math::ln64`
  instead of the platform `ln`: the filter size is written with the SSTable,
  so a platform whose `ln` differs in the last place could write a different
  file from the same entries. The fit error that ranks candidate models and
  the query variance square with an explicit multiplication instead of
  `powi(2)`. No stored bits change on aarch64-apple-darwin: the new golden
  digests are the same with the previous code, and filter sizes agreed on all
  21,299,890 `(n, rate)` pairs tried although the two logarithms differ in the
  last place on 1,220 of 74,693 sampled inputs
- `clippy.toml` `disallowed-methods` grows from 16 to the 50 entries
  `alice-zip` uses (inverse trigonometric and hyperbolic functions,
  `exp2` / `exp_m1` / `ln_1p` / `log` / `log10` / `cbrt` / `hypot`, and
  `powi`, whose multiplication order is unspecified)
- CI and `scripts/preflight.sh`: every test step (default / `ffi,sdf` / no default features) also requires `determinism_golden` to run and pass

## [0.3.0-beta.2] - 2026-10-08

### Added
- `law_store` module: stores `SignalLaw` (ALICE-Zip `law`, re-exported) under a name in either backend. `AliceDB::put_law` (next version), `get_law` / `get_law_version` (restored through `SignalLaw::from_parts`, which measures the residual again), `evaluate_law` (an `x` outside the valid range is refused, no extrapolation), `ingest_evidence` (judges new points, records the verdict with evidence count and RMS, and stores a parameter update as a new version while the judged version stays readable), `law_history`, `law_versions`, `law_names`. Records are checksummed blobs under the reserved key prefix `"\0alice-law\0"`; the layout is documented in the module, and damaged records, records copied under another key and records `from_parts` refuses are returned as errors (`LawStoreError`)
- `tests/law_store.rs`: 28 oracles with closed-form expectations (round trip in memory, through `to_bytes`, across close / reopen and blob compaction on the file backend, bit-identical parts; out-of-range refusal; each of the six verdicts recorded in order; versions kept after a parameter update; forged and damaged records; invalid names and policies; concurrent ingests)
- `examples/law_store.rs`, fuzz target `fuzz_law_record` (decoding arbitrary bytes as a law record)
- `law_store`: addressing a stored law by its content. `AliceDB::law_pointers_by_id` returns every `(name, version)` stored under a `SignalLaw::law_id`, and `AliceDB::evaluate_by_id` evaluates it — well defined for several pointers because equal identifier implies the evaluation returns the same bits (the converse does not hold, so an identifier is not a deduplication key). `law_id_key` builds the key; the pointer lives in the key, so one identifier addresses several laws without the record format holding a list and storing the same law twice is idempotent. Every path that writes a version writes the key, so the index cannot be complete for one entry point and missing for another
- `tests/law_id_index.rs`: 9 oracles (one identifier reached from two names evaluating bit-identically, round trip, the numeric semantics reaching the key, the version `ingest_evidence` writes being indexed too, survival through `to_bytes` / `from_bytes`, key families that cannot collide, an unknown identifier being empty rather than an error, and the key layout pinned by a golden)
- `README_JP.md` (renamed from `README.ja.md`), `scripts/docs_lint.py` + `scripts/test_docs_lint.py` (public-document vocabulary, CHANGELOG structure, README example = crate doctest), `scripts/stub_guard.sh`

- `segment::ResidualSemantics` with `residual_semantics` and
  `residual_is_applicable`: a residual blob says which arithmetic its model
  values came from. Blobs written before this version carry no identifier
  (`Unpinned`) and are still read, exact on the machine that wrote them and not
  checkable elsewhere; blobs written from now on carry it (`Pinned`). A pinned
  blob whose identifier does not match the arithmetic in use is **not applied**
  — the query paths return the model value rather than a corrupted one, and the
  mismatch is visible through `residual_semantics`
- `tests/law_single_source.rs`: the three read paths return the same bits for
  every analytic model over four lengths (2 / 5 / 65 / 129, chosen so that an
  integer timestamp maps to an integer sample position exactly), with a
  comparison-count gate and a per-path and per-model breakdown in the failure
  message
- `tests/legacy_segment_compat.rs`: segments serialised by the previous version
  (embedded byte fixtures, not re-generated here) still read back exactly, the
  fixtures are confirmed to carry the previous law's values, a freshly written
  residual names its arithmetic, old blobs report themselves as carrying none,
  and a residual naming a different arithmetic is skipped rather than applied

### Changed

- **Breaking:** `AliceDB::put_law` and `AliceDB::ingest_evidence` take the
  identifier of the numeric semantics the law is evaluated under
  (`&[u8; 32]`, as published by a deterministic-arithmetic crate). It is needed
  to compute the content identifier the version is indexed under, and it is a
  parameter rather than state on the database so that no path can store a
  version without indexing it
- **Breaking:** the three read paths now return the same bits. `generate_all`
  called the array generators in `alice-zip` while `query_point` and
  `query_range` called the point law, and the two disagreed: on the same
  segment 544 of 1206 compared samples came back with different bits
  (`SineWave` 168, `MultiSine` 181, `Fourier` 195; `query_range` and
  `query_point` agreed with each other, so `generate_all` was the one out of
  step, and it is `pub`). Every analytic model is now its law evaluated at
  `i = 0 ..= n-1` through the point law, which is also what `MemTable::seal`
  measures a residual against, so the stored data is anchored to the path that
  reads it. `generate_all` on a `Fourier` model returns `point_count` samples
  rather than `sample_count`
- **Breaking:** `segment::compress_residual_xor` takes the identifier of the
  arithmetic the model values were produced with (`&[u8; 32]`), and writes it
  into the blob. A residual stores `original ^ model`, so it only reconstructs
  the original when the reader evaluates the law with the same arithmetic;
  IEEE 754 leaves the transcendentals free to differ between targets, so
  without the identifier a reader cannot tell an exact reconstruction from a
  value that is neither the original nor the model
- `alice-zip` requirement raised to `0.7` so that the laws are evaluated
  through its point evaluators (`sine_at` / `multi_sine_at` / `fourier_at` /
  `polynomial_at`), whose transcendentals come from a deterministic-arithmetic
  crate rather than the platform libm. The local copies of the sine, Fourier,
  multi-sine and polynomial laws in `segment` are gone; `Linear` had two
  implementations as well (a reciprocal multiply in `generate_all`, a division
  in the point law) and now has one
- alice-zip 0.4 → 0.5.1 (no source change was needed; test results unchanged)
- The crate-level Quick Start is a compiled doctest on the memory backend (it was `ignore`d), and the crate doc states the dual license
- CI: test matrix adds `ubuntu-24.04-arm`; the law store oracles run on the memory backend alone; the example runs and benches compile on every OS; rustdoc also checks default features; docs lint on Linux / macOS / Windows. `scripts/preflight.sh` runs the same commands (`--quick` includes `cargo test --lib`; `cargo audit` keeps its database under the build directory)
- README rewritten from the source: what it is not for, installation with `cargo add`, the law store, storage backends; performance and compression figures without measurement conditions were removed

## [0.3.0-beta.1] - 2026-10-06

### Added
- `AliceDB::in_memory(StorageConfig)`: database held in process memory with the same write / read API as the file backend; one write sequence reads back bit-identically from either backend (`data_dir` is ignored, no WAL)
- `AliceDB::to_bytes` / `AliceDB::from_bytes(StorageConfig, &[u8])`: export / restore a whole database (segments + live blobs) as one CRC-32 checked buffer, from either backend; `to_bytes` flushes first (`alice_db::snapshot` documents the layout)
- `AliceDB::is_in_memory`, `StorageEngine::in_memory` / `from_snapshot_files` / `snapshot_files` / `is_in_memory`
- `wasm32-unknown-unknown` support with `default-features = false` (memory backend; `getrandom` uses its `js` backend and segment timestamps come from `Date.now()` on that target only)
- Test builds enable a new `open-file-audit` feature (through a self dev-dependency) that panics whenever the file backend renames over, deletes, or truncates through an append-only handle a file that Windows would refuse, so the Windows file rules are checked by every `blob_*` suite on Linux and macOS as well

### Changed
- **Breaking:** new default feature `fs` holds the file backend (`fs2`, `memmap2`): `AliceDB::open*` / `with_config*`, `compact_blob_sstable`, `compact_all_blob_sstables`, `blob_sstable_count`, `StorageEngine::new` / `open`, `SegmentView::open` / `open_read`, `SegmentSource::Mmap`, `DataSegment::write_rkyv`, `BlobStorage::open*` and the `blob_wal` / `blob_sstable` modules; default builds are unchanged, `default-features = false` users add `features = ["fs"]` to keep them
- `ffi`, `python`, `analytics`, `crypto` and `sdf` now enable `fs`
- `StorageEngine` with `enable_background_flush` returns the thread spawn error instead of panicking
- **License: `AGPL-3.0-or-later` → `AGPL-3.0-or-later OR LicenseRef-Commercial` (dual-licensed、2026-09-27)** AGPL 側の条件は変更なし (既存 AGPL 利用者への影響ゼロ)、商用という選択肢が追加されただけ SPDX が AGPL 単独だと cargo-deny / FOSSA / SBOM に「商用オプションなし」と見えるため宣言を dual に 変更点: SPDX / `LICENSE` → `LICENSE-AGPL` rename / `LICENSE-COMMERCIAL.md` (商用トリガー 6 条件 = クローズド製品・商用 SaaS・エッジ・ファームウェア配布・plugin 再配布・プラットフォーム NDA・保証、社内利用は AGPL 側で無償と明記) / README の選択肢表 商用窓口は法人 `contact@extoria.co.jp`

### Fixed
- Windows: opening a database whose blob WAL another handle holds returned the raw `os error 33` kind instead of `WouldBlock` (contention is now matched against `fs2::lock_contended_error()` on every platform; other I/O errors keep their kind)
- File backend: writing after reopening a database overwrote segments already on disk, because segment ids started again from 1 on every open (`seg_1.rkyv` was rewritten and its points lost). New ids now continue after the largest id in the loaded index (`tests/file_reopen_keeps_segments.rs`)
- Windows: every blob flush and compaction failed with `Access is denied` (os error 5). The blob WAL was opened in append mode, which on Windows grants `FILE_APPEND_DATA` but not `FILE_WRITE_DATA`, so truncating it after a flush failed; the WAL is now opened read + write and every append seeks to the end. Compaction in Overwrite mode also renamed the merged file over `blob.sst` while the store still had it mapped, and deleted superseded `SSTables` that a concurrent reader could still map, both of which Windows refuses. Readers now use mappings only under the `SSTable` list's read lock, and compaction drops every mapping under the write lock before it renames over or deletes any file. File names and the on-disk format are unchanged
- Blob compaction crash safety: superseded `SSTables` are deleted oldest first, compaction stops at the first failed delete (it used to log and continue), and the WAL is truncated only after every delete succeeded, so a delete that the merged file dropped can no longer be undone by a leftover older file. `.tmp` files left by a write that crashed before it was renamed into place are removed on open (`tests/blob_compaction_crash_safety.rs` interrupts compaction at each step and reopens)
- `scripts/run_tests.sh` counted no test of the required binary when cargo output is coloured (`CARGO_TERM_COLOR=always`), since a colour code follows `Running`; colour codes are now stripped before matching

## [0.2.0-beta.3] - 2026-09-17

### Added
- `tests/analytic_oracle.rs` — 閉形式 / 公開 test vector との突合 oracle 10 本: model 評価 (polynomial / linear / constant / sine / Fourier) の閉形式、`query_range` ≡ `query_point` ≡ `generate_all`、residual / RawLzma round-trip、**`lossless: true` の put → get bit 一致** (mmap / 非 mmap × polynomial / Fourier / raw)、既定 (lossy) fit の文書化閾値、aggregate 閉形式、CRC-32 check value + XXH64 test vector、Bloom filter の m / k 閉形式 + false negative 0 + FP ≤ 2p 不規則 timestamp の lossless は `#[ignore]` (下記)
- `segment::ResidualKind` / `residual_kind` / `apply_residual` / `compress_residual_xor` — XOR residual format (magic 0x02 LZMA / 0x03 raw)

### Fixed (oracle 先行 red 4 → 修正)
- **model 評価の法則が 4 箇所で手コピーされ、fitter (alice-zip) と食い違っていた**: 点 / 範囲 / archived 点 / archived 範囲の評価器が polynomial を `x = i/(n−1)` で評価 (係数は `fit_polynomial` の `x = i` 規約)、Fourier を raw DFT magnitude のまま加算 (`w·mag/n` 抜け = **n/2 倍**、1000 sample の sine が relative MSE 2.5e5 で復元)、sine の引数が `i/(n−1)` (generator は `i/n`) — `generate_all` (alice-zip 委譲) だけが正しく、lossless mode は同じ誤値に対する residual で隠していた → `segment::law` に 1 model 1 関数 (sample index `i` 引数) を置き 5 経路全部がそれを呼ぶ、`ModelType` doc に規約明記、旧 loop 群 (約 700 行) 削除 既存 `test_polynomial_segment` は x ∈ [0,1] 規約を pin していたので法則値 (i = 50 → 2500) に更新
- **`lossless: true` が bit 一致でなかった**: residual を `original − model` の f32 加算で持っていたため `model + residual` が丸める (−1.0500002 が −1.0500007) → residual を bit pattern XOR (`original ^ model`) に変更、既存 blob (additive、magic 0x00 / 0x01 / legacy) は読める

### Known limitation (format 変更が要るため未対応)
- `DataSegment` は timestamp を保持せず uniform spacing を仮定する: gap のある系列 (sensor drop-out) では residual index が別点に当たり lossless でも値が変わる、`scan` は存在しない timestamp を返す (`lossless_mode_is_exact_for_irregular_timestamps_too` が `#[ignore]` で記録)
- 既定 `FitConfig` は lossy (`lossless: false`) で、sine 系列は relative MSE < 0.1 (RMS で σ の 32 %) まで受容する — 文書化閾値どおりだが DB の既定として妥当かは未決定

### Fixed
- **FFI 14 関数の panic 隔離** (`src/ffi.rs`): `const fn` の `alice_db_version` を除く全 `extern "C"` を `ffi_guard(sentinel, || ..)` で包み、panic は host を落とさず sentinel (`DbResult::Unknown` / `PointResult { found: false }` / zero `DbStats` / null / −1 / false) + `alice_db_last_error()` (新規、`alice_db_clear_last_error` / `alice_db_free_error_string` も) で通知 `[profile.release] panic = "abort"` を撤去 (abort では `catch_unwind` が機能しない) release profile で guard test 通過

### Changed
- `alice-zip` 0.3 → 0.4: `PerlinNoise` segment / archived model は `generate_fbm_1d(n, seed, …)` (0.3 の `generate_perlin_advanced(n, 1, …)` と同一法則、sample 値は bit 一致) で再生成 0.4 は無効 parameter (`scale <= 0` / `octaves == 0`) を `Err` で返すため、corrupt model は corrupt blob と同じく zeros を返す `fit_polynomial` / `analyze_signal` / `generate_from_coefficients` / `generate_sine_wave` / `generate_multi_sine` / zlib wrapper は同名・同 convention (0.4 で Nyquist bin の復元重みが 2 → 1、energy threshold の母数が総エネルギーに変更されたため、新規 fit の係数選択が変わり得る 既存 segment の decode は互換)

## [0.2.0-beta.2] - 2026-09-15

### Added
- `ci.yml` (それまで fuzz / security-audit のみで test / clippy / doc は CI 未実行): test (default + `ffi,sdf`) / clippy `--all-targets -D warnings` 2 variant (pedantic は informational) / `msrv` job (`cargo +1.87 check`) / `feature-powerset` (cargo-hack depth 2) / fmt / doc `-D warnings` / actionlint、`alice-stubs` action、rust-cache
- `rust-version = "1.87"` を宣言 (alice-zip 0.3 の要求、`cargo +1.87 check` で実 compile 確認) + `rust-toolchain.toml` (1.98.1 pin)

### Fixed
- **mmap (zero-copy) 読み出しで `RawLzma` / `PerlinNoise` model の値が全て 0.0 になっていた** — `SegmentView::evaluate_archived_model` が「full deserialization が要る、production では compute する」という comment 付きで `0.0` を返す stub のまま publish されていた 手続き model に fit しない任意データは全部 `RawLzma` fallback に入るので、`put` → `flush` → `get` / `scan` が値を返さない (in-memory `DataSegment` 経路は正しかった) alice-physics 1.2.0 の `replay` / `db_bridge` 契約 test (`scan_positions` / `query_bodies`) が検出 (2026-09-15) archived 経路で LZMA 解凍 + dtype 再解釈 / Perlin 再生成を実装、range query は 1 回 decode、`raw_lzma_values_round_trip_through_flush_and_mmap` で pin
- **mmap 経路が lossless 残差 (`FitConfig::lossless`) を無視していた** — `SegmentView::query_point` / `query_range` (archived) が model 値だけを返し、in-memory 経路だけ残差を足していた 0.2.0-beta.2 で archived 経路も残差を適用、`raw_lzma_values_round_trip_through_flush_and_mmap` に lossless 2 次曲線 128 点の exact round-trip を追加
- clippy pedantic 13 件を 0 化し CI を `-W pedantic -D warnings` gate に (`sdf_bridge` の `# Errors` doc 7 / `ffi` の `let...else` 5 / float 比較 1)
- clippy `manual_range_contains` 3 件 (`sdf_bridge` test)、`incompatible_msrv` 1 件 (`segment.rs` の `File::lock_shared` は 1.89+ inherent、`fs2::FileExt::lock_shared` を trait 経由で明示)
- rustdoc `-D warnings` 5 件 (private item link / `blob::` 経由の誤った module path 3 / doc 例の非 ASCII byte string)

## [0.2.0-beta.1] - 2026-07-08

**β entry** for the `0.2.x` line. No code changes vs. `0.2.0-alpha.9`; runtime bytes are identical.

### Meaning of β

The α-3 blob KV programme is feature-complete as of `0.2.0-alpha.8` and dep-consumable as of `0.2.0-alpha.9`:

- Durable WAL (`0.2.0-alpha.2`).
- Batched fsync + cross-process advisory file lock (`0.2.0-alpha.3`).
- `SSTable` format v1 + WAL rotation (`0.2.0-alpha.4`).
- Multi-`SSTable` layout with full-merge compaction (`0.2.0-alpha.5`).
- Format v2 with per-`SSTable` Bloom filter (`0.2.0-alpha.6`).
- mmap read fast path + `LoadedSstable` / `BlobStorage` architectural split (`0.2.0-alpha.7`).
- Format v3 with persistent tombstones + heap-based `scan_prefix` (`0.2.0-alpha.8`).
- `alice_core` transitive dep switched to git for downstream consumability (`0.2.0-alpha.9`).

The β line is a **stability + performance focus**. Explicit β commitments:

- **No new public API surface**. Every new symbol lands in `1.0` or later.
- **No new on-disk format**. `SSTable` format v3 and the WAL record layout are the format contract for the `0.2.x` line. Any future format work ships as a separate `0.3.x` line, with a documented one-way migration path from v3.
- **Bug fixes ship as `0.2.0-beta.2` / `0.2.0-beta.3` / …**. A patch bump means "same API and format, safer implementation".
- **Perf work is welcome under β**. Better mmap advise, streaming k-way merge, per-op benchmarks, etc. — as long as the observable semantics stay the same.
- **`1.0.0`** ships after the β line proves stable against a downstream workload (currently ALICE-CodeTracker's `alice-tracker-alicedb` backend is the primary consumer).

### Changed

- `Cargo.toml`: `version = "0.2.0-alpha.9"` → `version = "0.2.0-beta.1"`.
- No other changes.

### Backward compatibility

- Every `0.2.0-alpha.9` test continues to pass unchanged (263 tests green).
- `SSTable` format v3 databases roundtrip identically.
- Downstream consumers (e.g. ALICE-CodeTracker's `alice-tracker-alicedb`) can point their git tag from `v0.2.0-alpha.9` to `v0.2.0-beta.1` with no other changes.

## [0.2.0-alpha.9] - 2026-07-08

Downstream-consumability fix — no runtime changes.

### Changed

- `alice_core` (the ALICE-Zip compression crate) is now a `git` dep on
  `https://github.com/ext-sakamoro/ALICE-Zip` instead of a
  `../ALICE-Zip/libalice` path dep. Consumers that pull `alice-db`
  from GitHub (e.g. `alice-tracker-alicedb`) can now resolve the
  transitive dep without needing a sibling ALICE-Zip checkout.
- A workspace-level `[patch."https://github.com/ext-sakamoro/ALICE-Zip"]`
  block redirects the git URL back to the local `../ALICE-Zip/libalice`
  path when the sibling checkout exists, so in-tree development still
  gets the live ALICE-Zip bytes.

Same pattern ALICE-CodeTracker adopted in its v0.4.0 shift from path
to git (`4daa0ef` — "fix(ci): resolve alice-zip via git dep instead of
local path").

### Backward compatibility

- No code, on-disk format, or API changes.
- Every alpha.8 test continues to pass unchanged (263 tests green).
- Consumers already on alpha.8 who build in-tree see identical bytes
  because the patch block resolves back to the same path dep.

## [0.2.0-alpha.8] - 2026-07-08

α-3.3b-3 subset: `SSTable` format v3 with tombstone-carrying records, closing the `FlushMode::Append` delete-doesn't-persist limitation carried since v0.2.0-alpha.5. Companion change: `scan_blob_prefix` is now a `BinaryHeap` k-way merge with priority-ordered newest-wins semantics.

### Added

- `FORMAT_VERSION_V3` (const `3`) and `VALUE_KIND_TOMBSTONE` (`0x02`).
  - v3 shares the on-disk layout with v2 (header + records + Bloom + 24-byte footer). Only the record encoding differs: tombstone records carry `value_kind = 0x02` and `value_len = 0`. The Bloom filter includes every key — tombstones included — so probes still work.
- `BlobSstable::write_from_iter` now accepts `BlobValue::Tombstone` entries and persists them as v3 tombstone records. The previous behaviour (reject with `InvalidInput`) applied when the format could not carry them; callers that fed a tombstone in expected an error, so this is technically a source-compatible loosening, not a break.
- `BlobSstable::open` / `LoadedSstable::open_mmap` accept v3 files (same 24-byte footer as v2) and expose `BlobValue::Tombstone` through `iter()` / `get()`.
- `LoadedSstable::get` returns `Some(BlobValue::Tombstone)` when the record is a tombstone, so `BlobStorage::get` masks correctly through the existing `Tombstone → Ok(None)` arm.
- 4 integration tests in `tests/blob_tombstone_persistence.rs`:
  - `FlushMode::Append` + delete + flush + reopen keeps the key deleted (this is the primary bug the release fixes).
  - `compact_all_blob_sstables` after a delete under `Append` folds the tombstone out — 1 file, no leftover tombstone.
  - `FlushMode::Overwrite` delete + reopen regression proof.
  - `Append` round-trip: put A → flush → delete A → flush → reopen → put A' → flush → reopen returns A'.
- 5 integration tests in `tests/blob_scan_kway_merge.rs`:
  - Multi-`SSTable` prefix scan returns sorted, deduped `(key, value)` pairs matching a hand-computed reference.
  - Memtable tombstone masks an older `SSTable` value.
  - Newer-`SSTable` tombstone masks an older-`SSTable` value.
  - Empty prefix → empty result.
  - Post-reopen (empty memtable) scan still surfaces every live key.
- Internal `SortedRun` helper in `blob.rs` — a peekable materialised sorted sequence, used as one arm of the k-way merge.

### Changed

- **`FlushMode::Append` now persists tombstones**. Previously, `flush_to_sstable(Append)` filtered tombstones out of the memtable, dropped them from memory, and truncated the WAL — meaning a delete followed by a flush and then a process restart resurrected the key from an older `SSTable`. As of v0.2.0-alpha.8 the whole memtable (including tombstones) is written into the new `SSTable` file. On reopen the tombstone surfaces through the sstable read path and correctly masks any stale value in an older file.
- `BlobStorage::scan_prefix` is now a k-way `BinaryHeap` merge across the memtable and each mmap'd `SSTable`. The heap is keyed on `(Reverse<Vec<u8>>, Reverse<usize>)` — smaller keys pop first, and within the same key the smaller source index (higher priority) wins. Older duplicates of the same key are drained after each winner emission. Complexity is O(N log R) where R is the number of sources (memtable + sstables). This replaces the O(N log N) `BTreeMap` accumulator used in v0.2.0-alpha.7.
- Every new `SSTable` write stamps `CURRENT_VERSION = FORMAT_VERSION_V3`. Existing databases from v0.2.0-alpha.2 through v0.2.0-alpha.7 open unchanged (their v1 / v2 files are still readable); the next flush or compaction upgrades their on-disk files to v3, at which point tombstones become persistent.
- `parse_records` and `build_offset_index` accept `VALUE_KIND_TOMBSTONE` and validate that its `value_len` is zero (fail-loud on corruption).
- The `expected 1 or 2` diagnostic on unknown version bytes now reads `expected 1, 2, or 3`.

### Backward compatibility

- Databases written by v0.2.0-alpha.2 through v0.2.0-alpha.7 open unchanged. Their `SSTable` files are v1 / v2 and continue to read via the same code paths.
- New writes are v3. Files that never receive a tombstone are byte-compatible with v2 readers *except for the version stamp* — an alpha.6 or alpha.7 process reading an alpha.8 file would fail with `unsupported sstable version 3`. This is a forward-compat break, not a backward-compat break; alpha.8+ processes read every prior format.
- Public API (`AliceDB` / `BlobStorage`) is source-compatible. `BlobSstable::write_from_iter` no longer errors on tombstone inputs — this is a source-compatible loosening for the one internal caller.
- `FlushMode::Overwrite` (the default) has identical semantics to v0.2.0-alpha.7. Only `FlushMode::Append` gains new behaviour (persistent tombstones).

### Design decisions

- **Same layout for v2 and v3**: keeping the 24-byte footer means `LoadedSstable::open_mmap` only needs to branch on the version stamp for the presence of a Bloom section; the offset-index build and Bloom parse are unified. This kept the α-3.3b-3 diff small.
- **Tombstones inside the Bloom**: keys that ever appeared in this `SSTable` — live or tombstoned — must be probeable, otherwise `get(key)` would fall through to older `SSTables` and see stale data. Including tombstones in the Bloom is a one-line change on the write side.
- **k-way merge over BTreeMap accumulator**: for scan workloads dominated by a small K (result size) and moderate R (sstable count), `log R` beats `log N`. The heap also gives a natural place to enforce newest-wins priority, whereas the `BTreeMap::insert` approach relied on iteration order.
- **Materialised `SortedRun` sources**: streaming iterators from `LoadedSstable::iter_prefix` across the heap would fight the borrow checker (each source needs `&self`, but the heap holds them alongside `winner_idx` indices). Collecting each source into a `Vec<(Vec<u8>, BlobValue)>` upfront trades some memory for simplicity; a fully streaming implementation is a future optimisation.
- **`Append`-mode compaction still drops tombstones**: `compact_all_sstables` merges every source into a `BTreeMap`, so a tombstone in a newer source correctly overrides a value in an older one, and the final `filter(!Tombstone)` drops both. Post-compaction files contain no tombstones — matching the LSM canonical.

### α-3 roadmap

α-3.3b-3 was the last step in the α-3 (blob KV) work. The blob store now supports:

- On-disk read fast path via mmap'd `SSTables` (α-3.3b-2).
- Bloom-guarded point lookups per `SSTable` (α-3.3b-1).
- Multi-`SSTable` accumulation and full-merge compaction under `FlushMode::Append` (α-3.3a).
- Batched fsync policies and cross-process file locks (α-3.1).
- Durable WAL persistence (α-3.0 / α-2 / α-alpha.2).
- Persistent tombstones and a heap-based `scan_prefix` (this release).

The α-4 line — starting with the ALICE-CodeTracker `alice-tracker-alicedb` backend, then benchmarks, then a formal β entry — is now unblocked.

## [0.2.0-alpha.7] - 2026-07-08

α-3.3b-2 subset: on-disk read fast path via memory-mapped SSTables and the `BlobStorage` architectural split.

`BlobStorage` now separates its state into a `memtable` (recent writes replayed from the WAL) and a newest-first `Vec<Arc<LoadedSstable>>` (settled state, held via mmap). Reads consult the memtable first, then fall through to each SSTable in order, using the Bloom filter shipped in v0.2.0-alpha.6 to reject absent keys before touching the mmap. Only key offsets stay in memory; the value bytes remain on disk until a read actually copies them out. Existing databases (α-3.2a onwards) open unchanged.

### Added

- `blob_sstable::LoadedSstable` — mmap-backed `SSTable` reader.
  - `open_mmap(path)` — walks the records section once to verify per-record CRCs and build the key → offset index. Missing files return `Ok(None)` so a fresh directory opens cleanly.
  - `get(key) -> Option<BlobValue>` — Bloom-guarded point lookup; returns `None` without touching the index when the Bloom rejects. On a hit, copies the value bytes out of the mmap into an owned `BlobValue`.
  - `iter()` / `iter_prefix(prefix)` — ordered walks used by compaction and `scan_blob_prefix`.
  - `bloom()` / `path()` / `len()` / `is_empty()` accessors.
  - `Debug` impl surfaces the summary (path, mmap length, Bloom presence, record count) without dumping the raw bytes.
- `blob_sstable::build_offset_index` — internal helper that walks the records slice and returns a `BTreeMap<Vec<u8>, RecordOffset>` (private).
- 6 unit tests in `src/blob_sstable.rs::tests` covering: raw value round-trip, compressed value round-trip verbatim, absent-key resolution, full-range iteration, prefix-range iteration, legacy v1 open path (hand-written to alpha.5 bytes).
- 8 integration tests in `tests/blob_mmap_read.rs`:
  - Flush + reopen serves both `Raw` and `Compressed` values through the mmap path.
  - `FlushMode::Append` multi-`SSTable` reads return newest-wins across two flushes.
  - A memtable tombstone masks a live value in an older `SSTable`, both live and after reopen (WAL replay).
  - Bloom-guarded absent keys return `None` (functional coverage; a perf benchmark is out of scope for α-3.3b-2).
  - `scan_blob_prefix` merges memtable and multiple `SSTables` with newest-wins semantics and stable ascending order.
  - `compact_all_blob_sstables` after four `Append` flushes folds every live key into one file, all values preserved.
  - 8-thread concurrent read/write against a shared store observes a consistent view.
- `memmap2 = "0.9"` dependency (was already declared for a future migration; now actively used).

### Changed

- `BlobStorage` fields:
  - `inner: Arc<RwLock<BTreeMap<Vec<u8>, BlobValue>>>` (which held every live value in memory) is replaced by
  - `memtable: Arc<RwLock<BTreeMap<Vec<u8>, BlobValue>>>` (WAL-replayed diff since the last flush) and
  - `sstables: Arc<RwLock<Vec<Arc<LoadedSstable>>>>` (newest-first).
- `BlobStorage::open_with_config`: enumerates every `SSTable` on disk, opens each via `LoadedSstable::open_mmap`, and reverses the list to newest-first. The WAL replay populates only the memtable — record values living inside `SSTables` stay on disk until a read touches them.
- `BlobStorage::get`: memtable first, then iterates `sstables` newest-first calling `LoadedSstable::get`. Snapshots the sstable list (a cheap `Vec<Arc<_>>` clone) so the read lock does not span the probes.
- `BlobStorage::put` / `BlobStorage::delete`: mutate the memtable only. The WAL still records every change so a reopen reconstructs the same memtable state.
- `BlobStorage::flush_to_sstable`:
  - `FlushMode::Overwrite`: delegates to `compact_all_sstables` — the merged file replaces every prior `SSTable` and the memtable clears.
  - `FlushMode::Append`: writes the memtable's non-tombstone entries to a fresh sequentially-numbered `SSTable` file, mmap-opens it, prepends it to the sstable list, prunes tombstones from the memtable, and truncates the WAL. Semantically identical to v0.2.0-alpha.5.
- `BlobStorage::compact_all_sstables`: folds every mmap'd `SSTable` (walking `sstables` in `rev()` for oldest-first order) plus the memtable into a merged `BTreeMap`. Tombstones drop out safely because we rewrite every file in one operation. The new file is written to a `.tmp` sibling, atomically renamed into place, mmap-opened, and swapped into `sstables` **before** the older files are unlinked. On Unix an unlink under a live mmap is safe; on Windows the deletion may fail and is logged rather than propagated, leaving stragglers to be absorbed by the next compaction.
- `BlobStorage::scan_prefix`: merges memtable and every `SSTable`'s prefix window into a `BTreeMap` accumulator with newest-wins semantics (`sstables.iter().rev()` for oldest-first, then memtable on top). Materialises values (decompressing where needed) and skips tombstones.
- `BlobStorage::len`: reuses the merge approach — walks every source without decompressing and counts non-tombstoned survivors.
- `BlobStorage::sstable_count`: now reports the length of the loaded list (`self.sstables.read().len()`) rather than a directory scan; the signature retains `io::Result` so callers do not need to change.

### Backward compatibility

- Existing databases (v0.2.0-alpha.2 through v0.2.0-alpha.6) open unchanged. `LoadedSstable::open_mmap` accepts both format v1 and v2 files.
- `FlushMode::Overwrite` (the default) preserves v0.2.0-alpha.5 semantics exactly: every flush rewrites `blob.sst` and clears the memtable.
- `FlushMode::Append` preserves v0.2.0-alpha.5 semantics including the pre-existing limitation that a delete-then-flush cycle cannot survive a process restart (see below). No new tombstone bug is introduced.
- Public API (`AliceDB::open` / `open_with_blob_config` / `get_blob` / `put_blob` / `delete_blob` / `scan_blob_prefix` / `flush_blobs` / `compact_blob_sstable` / `compact_all_blob_sstables`) is source-compatible.

### Known limitation carried forward

`FlushMode::Append` cannot persist tombstones across a process restart: the `SSTable` format has no tombstone record, so a `delete_blob` followed by a flush and then a reopen may resurrect the value from an older `SSTable`. Same behaviour as v0.2.0-alpha.5. Workarounds:

- Call `compact_all_blob_sstables` after any batch of deletes under `Append`; the merged file drops tombstoned keys.
- Switch to `FlushMode::Overwrite` (the default) if durable deletes are required.

α-3.3b-3 lifts this limitation by extending the `SSTable` format to carry tombstone records.

### Design decisions

- **Key offsets in memory, values on disk**: the `LoadedSstable` index holds one entry per record (`BTreeMap<Vec<u8>, RecordOffset>`), where `RecordOffset` is 32 bytes. Value bytes stay in the mmap and are copied into an owned `BlobValue` only when a read materialises them. For workloads with large values (e.g. compressed `Stub` JSON in ALICE-CodeTracker) this is the significant memory savings.
- **Newest-first sstable list**: reads walk the list in stored order and return as soon as any `SSTable` resolves the key. Compaction visits the list in `rev()` for merge semantics. Keeping newest-first matches the read-path priority and avoids repeated reversals.
- **Bloom-guarded lookup at the SSTable layer**: `LoadedSstable::get` probes the Bloom first (when present) before touching the index. For v1 files (`bloom() == None`) the Bloom is treated as always-accept — correctness is preserved, no acceleration.
- **`Arc<LoadedSstable>` for cheap sharing**: `get` clones the sstable list under the read lock and drops the lock before probing. Cloning a `Vec<Arc<_>>` is O(N) reference bumps, cheap enough for the alpha and future concurrency stories.
- **Swap-then-delete for compaction**: the merged file is materialised and swapped into `sstables` before old files are unlinked. On Unix this is safe against live mmaps (the inode outlives the directory entry). On Windows a still-mapped file cannot be deleted; we log and continue rather than fail the whole compaction, and the next compaction absorbs the leftovers.
- **`scan_prefix` accumulates into a `BTreeMap`**: correct newest-wins semantics with tombstone masking, at the cost of O(N log N) merging. α-3.3b-3 introduces a sorted-run merge iterator that avoids the accumulator.

### α-3 roadmap remainder

- **α-3.3b-3**: sparse index / prefix trie inside each `SSTable` for accelerated `scan_blob_prefix` + tombstone-carrying `SSTable` format extension to lift the `Append`-mode delete limitation.

## [0.2.0-alpha.6] - 2026-07-08

α-3.3b-1 subset: durable investment for the on-disk read fast path. Introduces `SSTable` format v2 with an embedded Bloom filter per file and a longer 24-byte footer (`records_size` + `bloom_size` + magic). Format v1 files written by v0.2.0-alpha.2 through v0.2.0-alpha.5 continue to open and are transparently upgraded to v2 on the next flush or compaction. The read path itself continues to load records into memory — mmap + Bloom-guarded point lookups land in α-3.3b-2.

### Added

- `src/bloom.rs` — dependency-free Bloom filter.
  - `BloomFilter::with_capacity(expected_elements, false_positive_rate)` sized via `m = ⌈-n·ln(p)/(ln 2)²⌉`, `k = ⌈m/n·ln 2⌉`.
  - `BloomFilter::from_raw(bits, num_bits, num_hashes)` reconstructs a filter from its serialised components (used by the SSTable reader).
  - `insert` / `contains` implemented via Kirsch-Mitzenmacher double hashing over `std::hash::DefaultHasher` (SipHash-2-4), so the crate remains dependency-free.
  - 5 unit tests: empty filter rejects, no false negatives, false-positive rate stays within 2× of target at n = 4 096 keys, roundtrip through raw bits preserves membership, sizing scales with element count.
- `blob_sstable::BlobSstable::bloom()` accessor — returns `Some(&BloomFilter)` for v2 files and `None` for legacy v1 files.
- `blob_sstable` constants: `FORMAT_VERSION_V1` / `FORMAT_VERSION_V2` / `CURRENT_VERSION` / `FOOTER_LEN_V1` / `FOOTER_LEN_V2` / `BLOOM_HEADER_LEN` / `BLOOM_FALSE_POSITIVE_RATE`.
- 4 integration tests in `tests/blob_sstable_bloom.rs`:
  - New v2 writes carry a Bloom filter that accepts every inserted key (no false negatives).
  - Bloom rejects clearly-absent keys (≥ 100 / 128 unrelated probes rejected).
  - Legacy v1 files (hand-written to alpha.5 bytes) still open; `bloom()` reports `None`; records read back correctly.
  - `AliceDB::open` on a directory with a legacy v1 `blob.sst` reads it transparently; the first `compact_blob_sstable()` upgrades it to v2.

### Changed

- `BlobSstable::write_from_iter` now:
  - Builds a Bloom filter during record iteration (sized by observed record count at the classic 1 % false-positive rate).
  - Writes the Bloom section between the records and the footer.
  - Emits a v2 footer: `records_size (u64) + bloom_size (u64) + magic (8)` — 24 bytes instead of the v1 16.
  - Header now stamps `FORMAT_VERSION_V2`.
- `BlobSstable::open` branches on the header version:
  - v1: reads the 16-byte footer as `records_size + magic`; `bloom()` returns `None`.
  - v2: reads the 24-byte footer as `records_size + bloom_size + magic`; parses the Bloom section and stores the reconstructed filter.
- Any newly-written or newly-compacted SSTable is v2. Read-side callers that need to distinguish should probe `.bloom().is_some()`.

### Backward compatibility

- v1 files (`ALICEBBS` header + 16-byte footer) written by any of v0.2.0-alpha.2 through v0.2.0-alpha.5 continue to open on `BlobStorage::open_with_config` and `AliceDB::open`. The reader simply reports `bloom() == None` — the read path already probes the record map, so a missing Bloom is a semantic no-op (equivalent to a filter that never rejects).
- The first flush or compaction on such a database rewrites the file as v2 with a Bloom filter attached. No user-visible action is required.

### Design decisions

- **Bloom sized by observed record count**: the writer knows the exact record count once it has iterated the input, so it sizes the filter for that count at the fixed 1 % target rate. This keeps every file's on-disk overhead proportional to its own contents (~1.2 bytes/key).
- **`std::hash::DefaultHasher` for hashing**: SipHash-2-4 is the standard library's built-in hasher, so we avoid adding a hashing dependency at this stage. `α-3.3b-2` can swap in `xxhash-rust` or `wyhash` if benchmarks justify it.
- **Kirsch-Mitzenmacher double hashing**: one SipHash call per key produces two `u64` values (via a differentiating suffix on the second call); the `k` bit positions are derived as `h1 + i · h2 mod m`. This preserves the standard false-positive-rate analysis while doing only one hash call per operation.
- **CRC32C on the Bloom section**: same integrity story as records. Corrupted files fail loudly on open.
- **Read path unchanged in α-3.3b-1**: the Bloom filter is *loaded and available* but not yet consulted by `get_blob`. That gates on the mmap + `LoadedSstable` split that lands in α-3.3b-2; shipping the format now means α-3.3b-2 does not need a second format migration.

### α-3 roadmap remainder

- **α-3.3b-2**: mmap-backed read path (`LoadedSstable` = `memtable` + `Vec<mmap SSTables>`) + Bloom-guarded point lookups on `get_blob`. The format v2 infrastructure shipped here is a prerequisite.
- **α-3.3b-3**: sparse index / prefix trie inside each SSTable for accelerated `scan_blob_prefix`.

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
