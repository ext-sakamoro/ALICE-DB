# Changelog

All notable changes to ALICE-DB will be documented in this file.

## [Unreleased]

### Added
- CI (ALICE-Law G9): `scripts/id_move_check.py` (ALICE の識別子を持つ crate で共有、ALICE-Zip と同じ file) を追加 最新の release tag の公開版と作業 tree で `DB_SEMANTICS_ID` と residual / law record に刻む `SEMANTICS_ID` の値を読み、動いたのに破壊的な版になっていない / 公開版の値が以前の識別子の一覧に無い / CHANGELOG が両方の値を書いていない時に fail する 型が `[u8; 32]` の定数は全部、検査対象か理由付きの除外に載せる (ci.yml の clippy job と preflight)
- `alice_db::DB_SEMANTICS_ID` (`src/semantics.rs`、ALICE-Law G6): この crate 自身の算術の識別子 `tests/determinism_golden.rs` が bit を記録する 5 場面 (model の選択 / `query_point` と `query_range` / 集計 / bloom filter の大きさ / 法則の識別子と評価) の SHA-256 と `alice_det_math::SEMANTICS_ID` を、`alice-det-math` と同じ手順 (名前順、u32 big-endian の名前長、名前、digest 32 byte を 1 つの SHA-256) で畳んだ値 (alice-det-math 0.5.1 で `aab1eb3d…51162f`) 表 `DB_SEMANTICS_BY_DET_MATH` に行の無い `alice-det-math` の世代では compile が止まる `DB_SEMANTICS_CHECKED_AT` は pin を最後に確かめた crate の版と toolchain で、`Cargo.toml` の版か `rust-toolchain.toml` の channel が違うと試験が落ちる 試験 `tests/db_semantics_id.rs` (9 件) は各 entry を golden の試験の source から読んで比べ、記録した golden がすべて畳まれていること、実行中の rustc の版も確かめる 変異 (golden を変えずに entry を変える / entry を変えずに golden を変える) で red を確認した crate root の `SEMANTICS_ID` (算術の識別子、`alice-det-math` の値) は変わらない
- `law_store`: 法則の記録の読み出しが記録の識別子を確かめた結果を返す型 `RecordSemantics` (`Current` / `VerifiedPrevious(id)` / `Unpinned`)、`StoredLaw` / `LawValue`、方針 `LawReadPolicy` (`AcceptUnpinned` が既定、`RequirePinned` は識別子の無い記録を拒む) と `get_law_with` / `get_law_version_with` / `evaluate_law_with` / `evaluate_by_id_with`、error `LawStoreError::LawRecordMismatch { stored, running }` (両方の識別子を表示) と `LawStoreError::Unpinned` 試験 `tests/law_record_semantics.rs` (11 件、0.5 の encoder が書いた形式 1 の記録を fixture として持つ 以前の識別子 `135c407d…` (alice-det-math 0.5.0) で書いた記録は `VerifiedPrevious` で読める)
- CI: `clippy.toml` の禁止が外部 crate の関数を名指す場合に、その module の公開関数がすべて禁止されているか理由つきで許可されているかを確かめる `scripts/clippy_external_coverage.py` (試験 `scripts/test_clippy_external_coverage.py`) を足した 公開関数は `Cargo.lock` の版の rustdoc JSON (固定した toolchain で `RUSTC_BOOTSTRAP=1`) から、この workspace が全 feature で解決する feature の組で `#[doc(hidden)]` の関数も含め、`pub use` の再公開を辿って読み、JSON の形式の版が想定外なら止まる 禁止も許可もされていない公開関数・存在しない許可・禁止と許可の重複・理由の無い許可・版上げで消えた禁止は失敗にする 許可は `scripts/clippy_external_allow.toml` に書く この crate の `clippy.toml` は外部 crate を名指さないので、今は比較する対象が無いことを報告して通る clippy job と preflight で走る

- CI (ALICE-Law G14): 浮動小数点の結果を変える build の flag を、静的と動的の 2 段で検査する `scripts/build_config_guard.py` (静的) は target/ 以外の `.cargo/config*`・`*.toml`・`build.rs`・`*.rs`・`*.yml`・`*.json`・script 類と rustc wrapper を comment を除いて読み (TOML / JSON は decode 後の値も)、`target-cpu=native`、素の literal でない target-cpu / target-feature / llvm-args の値 (変数・escape・引用の継ぎ・連結・format)、command や file から取った RUSTFLAGS、LLVM と C の fast-math 系の flag、sse / sse2 を外す target-feature、`src/` の fast 系 float intrinsic を前後の文脈によらず検出する 意図的な出現は `scripts/build_config_allowlist.toml` に理由付きで書き (allow は行の完全一致と件数、build が読む file の exclude は拒否) `scripts/build_flags_check.py` (動的) は新しい target dir で `cargo check -v` を走らせ、cargo が rustc に実際に渡した flag を読む (rustc の command line が 1 つも無い時、この crate のものが無い時は失敗) 静的は docs-lint job (ubuntu / macOS / Windows) と preflight、動的は release-profile job (ubuntu / macOS、default / all-features / no-default) と preflight で実行
- `tests/float_environment.rs`: 丸めが round-to-nearest-even であること (f64 / f32 の tie の両方向) と、subnormal が 0 に flush されないこと (結果と入力の両方) を実行時に確かめる test test job の全 `run_tests.sh` step (4 OS) と release profile の step で必須にした `alice-law-conformance.toml` の G14 を pass にした

### Changed

- CI (ALICE-Law G5): release-profile.yml の各脚が ci.yml の test job と同じ必須 binary を数えることを `scripts/release_lists_check.py` で確かめる (cargo の引数で対にし、相手の無い step、必須の一覧の違い、比較 0 件は失敗) 試験 `scripts/test_release_lists_check.py` docs-lint job と preflight で走る
- **破壊的:** blob の SSTable を形式 v4 にした Bloom filter の位置を std の `DefaultHasher` (release 間で algorithm を固定しない) から crate 内の SipHash-2-4 (`bloom::siphash24`、参照 test vector と std の `SipHasher` で確認) に移し、Bloom 節に位置の導出の識別子 `bloom::BLOOM_HASH_ID` (説明文 `BLOOM_HASH_DESCRIPTION` の SHA-256) を持たせた (ALICE-Law G8 / G10) 読みは識別子が一致する filter だけを使い、v2 / v3 の filter と別の識別子の filter は file の key から作り直して `bloom_source()` (`BloomSource::Rebuilt`) で示す 旧 filter の位置を黙って使うと toolchain の更新で存在する key を「無い」と答えうるため 位置は試験で固定 module doc の hash の記述も実装に合わせた 試験 `tests/bloom_fixed_hash.rs` (v3 の実 file を作り直して 301 key で偽陰性 0、新 file は保存の filter、別の識別子は作り直し)
- ALICE-Law G2: `clippy.toml` が f32 / f64 の `max` / `min` / `clamp` を全 target で禁止し、残っていた箇所 (compaction の score、memtable の fit 誤差、bloom の大きさ、segment の標本 index、sdf_bridge の Morton 格子、試験の参照計算) を `alice_det_math` の `max` / `min` / `max64` / `min64` / `metric::clamp` に移した 結果の bit は変わらない (golden と float edge の試験は debug / release とも無変更で通る) canary の表は 64 行 conformance の G2 を pass に (tie_tests は集計の Min / Max と group の running min / max、他の箇所は整数への変換か非負の値で tie が公開 API に現れない理由を記載)
- CI: README の ALICE-Law 節の検査 (law-readme-block.yml) を preflight でも同じに走らせる `scripts/law_readme_block_check.py` を足した 節の pin の ALICE-Law の生成器を、workflow の check と同じ引数 (`--crate` / `--law-main` / `--lint` は workflow から読む) で、fetch した ALICE-Law の clone (`ALICE_LAW_DIR`、無ければ job と同じ URL の blob-less clone を cache に作る) に対して走らせる 節も workflow も無ければ n/a、workflow があって節が無い・節があって workflow が無い・比較 0 件・件数の行が無い・pin が clone に無い・fetch の失敗は失敗 これまで conformance の宣言を変えても preflight は通り、CI の "ALICE-Law README block" だけが落ちた (G8 の branch で実測) 試験 `scripts/test_law_readme_block_check.py` (12 件、偽の ALICE-Law で各結果を作る) は docs-lint job (3 OS) と preflight で走る 共有 file (README の節を持つ repo に同じ中身で置く)
- CI (ALICE-Law G5): release-profile.yml が release profile で試験全体を走らせる ci.yml の test job の各脚 (既定、全 native feature、既定なし、memory backend の law_store) を同じ必須 binary と件数で `--release` 付きで走らせる (ubuntu / macOS aarch64) これまでは golden 2 file と float_environment だけで、release だけで変わる Min の畳み込みを見逃した (変異で実測) preflight の full も同じ conformance の G5 を pass に
- CI: clippy の canary の表の生成器 `scripts/gen_clippy_canary_calls.py` が generic 関数 (`std::thread::spawn` / `scope` / `Builder::spawn` / `spawn_scoped`) を turbofish 付きで植えるようにした (素の path では型が決まらず compile できないため) 他の行は byte 一致で、この repo の表は変わらない 共有 file の協調更新 (ALICE-Analytics などと同じ中身)
- **破壊的:** 法則の記録 (`law_store`) を形式 2 にした (ALICE-Law G7 / G8) 記録は書いた時の算術の識別子 (`put_law` / `ingest_evidence` に渡す `semantics_id`) 32 byte を版の直後に持ち、読み出しは実行中の `SEMANTICS_ID` と比べる 同じなら `Current`、`alice_det_math::check_semantics` が同じ bit を再現すると確かめた以前の識別子なら `VerifiedPrevious`、それ以外は `LawRecordMismatch` で断る (識別子の不一致を識別子の無い記録として扱うことはしない) 0.5 以前の形式 1 の記録は従来どおり読め、結果は `Unpinned` になる `AliceDB::to_bytes` / `from_bytes` で別の build に渡した記録も同じ検査を通る これまで記録に識別子が無く、別の算術で書いた法則を読み手の算術でそのまま評価していた
  - 公開 API: `get_law` / `get_law_version` は `Option<SignalLaw>` から `Option<StoredLaw>` (`.law` と `.semantics`)、`evaluate_law` / `evaluate_by_id` は `f64` から `LawValue` (`.value` と `.semantics`) を返す `encode_law_record` は 4 番目の引数 `semantics_id`、`decode_law_record` は 4 番目の要素 `Option<[u8; 32]>` (形式 1 は `None`) を持つ `LAW_RECORD_FORMAT` は 1 から 2 判定の記録も形式 2 にした (下の項)
  - 利用者調査 (ALICE-* の main): `alice-db` を依存に持つ 11 repo (版指定 9、path 2) のどれも `law_store` の API を呼んでいない 版を 0.5.1 から 0.6.0 に
- **破壊的:** 0.3.0 前の残差 (算術の識別子を持たない lossless 残差、`ResidualSemantics::Unpinned`) を既定の読みで適用しない (ALICE-Law G8) `ResidualPolicy::Refuse` (既定) と `DataSegment` / `SegmentView` の `query_point` / `query_range` は `ResidualError::Unpinned` (segment、key の範囲、store を表示) で断る 適用するのは明示の `ResidualPolicy::AcceptUnpinned` / `UnpinnedResiduals::Accept` の時だけで、結果の状態は `ReadStatus::Unpinned` になる `residual_is_applicable` は識別子の無い残差に `false` を返す これまで識別子の無い残差は、書いた機械と算術が違っても黙って適用されていた
  - 読みの結果に状態を持たせた: `ReadStatus` (`Current` / `VerifiedPrevious(id)` / `Unpinned` / `ModelOnly`) を `DataSegment` / `SegmentView` / `StorageEngine` の `read_point` / `read_range` と `AliceDB::get_with_status` / `scan_with_status` が返す 範囲の読みは、返した点の segment の状態のうち最も弱いもの (`Current` < `VerifiedPrevious` < `Unpinned` < `ModelOnly`) memtable の値は `Current`、`ResidualPolicy::ModelOnly` の segment は `ModelOnly`
  - 試験 `tests/residual_unpinned_store.rs` (5 件) は 0.3.0-beta.1 (`52ca813`) が書いた store (`tests/fixtures/store_pre_0_3_0`) を読み、既定で断ること、`AcceptUnpinned` で 128 値が bit 一致し状態が `Unpinned` であること、範囲の状態の集約を見る `tests/legacy_segment_compat.rs` の旧 segment の試験は既定の拒否と opt-in での復元の両方を見る
  - 利用者調査 (ALICE-* の main): `ResidualPolicy` を使うのは ALICE-Edge の `db_bridge` (`Refuse` / `ModelOnly`、alice-db 0.5 指定) のみで、0.3.0 前の store を読む利用者は無い
- **破壊的:** 判定の記録 (`law_store` の verdict) を形式 2 にした (ALICE-Law G8) 記録は判定した build の `DB_SEMANTICS_ID` 32 byte を名前の直後に持つ (RMS と結果はこの crate 自身の算術のため) `AliceDB::law_history_with(name, policy)` は `StoredVerdict { record, semantics }` を返し、この build の識別子なら `Current`、別の識別子は `LawStoreError::VerdictRecordMismatch { stored, running }` で断る (同じ判定を計算する以前の `DB_SEMANTICS_ID` は定めていない) 形式 1 の記録は `Unpinned` で読め、`LawReadPolicy::RequirePinned` では `LawStoreError::Unpinned` で断る `law_history` は従来どおり `Vec<VerdictRecord>` を返す (`AcceptUnpinned`) `VERDICT_RECORD_FORMAT` は 1 から 2 試験 `tests/verdict_record_semantics.rs` (3 件、main 66dbb8e が書いた形式 1 の記録 2 件を bytes で持つ)
- CI: 公開 API の互換性検査を、直前の commit (`HEAD~1`) との比較で結果を問わない informational な job から、crates.io の最新の alice-db との比較で失敗させる `semver-checks.yml` に置き換えた `scripts/semver_gate.py` が consumer の feature の組 (既定 / 既定なし / `.github/semver-gate.toml` の組、ALICE-Law の release-train.toml と同じ) ごとに `cargo semver-checks --baseline-version <最新>` を走らせ、Cargo.toml の版との差 (0.x では minor が major) が許さない変更、tool が走らない、別の step を名乗る、比較 0 件のいずれでも失敗する major の step は比較しない 公開版が無い crate は理由を出して通す Cargo.toml が公開済みの版と同じ時は次の最小の版 (0.5.1 なら 0.5.2) として検査し、`--release-type` を明示する その時は検査する版を `alice-db 0.5.1 equals the latest release 0.5.1: checking it as the next release 0.5.2 (minor step)` の形で出す (cargo-semver-checks 0.50 は現在の版を指定できず、tool 自身の行は `v0.5.1 -> v0.5.1` のまま) 公開版より古い版は拒否する 走るのは `src/**` / `Cargo.toml` / `Cargo.lock` / 検査器と workflow の変更時と週 1 回 試験 `scripts/test_semver_gate.py` は記録した tool の出力と、実物の cargo-semver-checks を 2 つの小さな crate に走らせる試験を含む preflight の full にも同じ step を足した 判定の関数は ALICE-Law `scripts/release_train.py` と同じ名前と結果で、2 file は repository 固有の名前を持たない
- CI: `scripts/semver_gate.py` の feature の全体を `cargo metadata --no-deps` から読むようにした これまで consumer の組を宣言しない crate の「全 feature」の組は manifest の `[features]` の key だけで、どの feature からも名指されない optional 依存の暗黙の feature (`#[cfg(feature = "<依存名>")]` の下の公開 API) は検査されなかった cargo metadata の失敗、JSON でない出力、package が見つからない場合は exit 2 で、`[features]` の key に戻すことはしない workspace では package を名前で選ぶ 宣言した組が名指してよい名前も同じ全体 (暗黙の feature を含む) にした 組の網羅を見る検査が同じ名前の集合を使えるように、feature が有効にする feature の閉包 (`feature_closure`) と、どの組からも届かない feature (`uncovered_features`)、未定義の名前 (`unknown_features`) の関数を足した DB の組 (既定 / 既定なし / 宣言した 2 組) は変わらず、全体は alice-analytics / analytics / crypto / ffi / fs / numpy / open-file-audit / pyo3 / python / sdf で、どの feature も宣言した組の閉包に入る 試験は実物の cargo metadata で暗黙の feature、`dep:` で名指した依存、閉包、workspace、失敗を見る 11 件と、閉包を素の名前にする変異と全体を `[features]` の key にする変異で落ちることの確認、名指されない optional 依存の下にだけある関数を minor の版で消すと全 feature の組で拒否される実物の cargo-semver-checks の試験を含む
- CI: `scripts/semver_gate.py` が宣言した consumer の組の網羅を検査するようにした これまで CLI は feature の全体を出すだけで `uncovered_features` を呼ばず、後から足した feature はどの組にも入らないまま build も検査もされず exit 0 だった `.github/semver-gate.toml` が組を宣言する時、全体の feature がどの組の閉包にも `exclude-features` にも無ければ `feature <名前> is in no consumer-features / consumer-check-features set and not in exclude-features` を feature ごとに出して exit 1 にする 組を宣言しない時は全 feature の組が全体なので検査は不要で、その旨を出す `exclude-features` は ALICE-Law の roster と同じ形 (`feature = "build しない理由"` の table、理由は空でない) で、組の宣言が要り、全体に無い名前・`default`・組にも入っている名前は設定の誤り (exit 2) DB の全体 10 件は宣言した組の閉包に入るので通り、`zz = []` を足すと exit 1 になることを確かめた
- CI: `scripts/semver_gate.py` が cargo-semver-checks の出力を読む parser であることを `CARGO_OUTPUT_PARSER = True` で宣言し、`scripts/test_semver_gate.py` は module を `import semver_gate` で読むようにした (色の付いた出力を試験する検査器が parser とその試験を見つけられる形 実行時は従来どおり `CARGO_TERM_COLOR=never` と escape の除去) 共有 file の中身を揃える変更で、検査の挙動は変わらない
- CI: clippy の禁止一覧の canary (`scripts/clippy_canary.py`) が `clippy.toml` の全 entry を植えて確かめるようにした 植える呼び出しの表 `scripts/clippy_canary_calls.tsv` は `scripts/gen_clippy_canary_calls.py` が `clippy.toml` から 1 entry 1 行で生成し (この repo は 58 件、以前は手書きの 7 件)、CI が再生成して byte で比べる canary は全 entry に行があること (植えるか、理由付きで飛ばすか) を確かめ、外部 crate の path は `Cargo.lock` の版を固定して依存に足す 型も禁止された inherent method (`RandomState::new`) は、その型の診断を 1 件だけ許す
- CI: `scripts/run_tests.sh` が名前ごとの最低の件数 (`float_environment=2`) を受け付けるようにし、G14 の `tests/float_environment.rs` を名指す全 step を `float_environment=2` にした 2 つの試験の片方を `#[ignore]` にする・cfg で外す・名前を変えると、その step が失敗する 名前だけの指定は従来どおり 1 件以上 `=0` や数でない値は command を走らせる前に exit 2
- CI: 公開する文面の検査の pin を ALICE-Law 779fa44 (checker sha256 e439fdd7…) に進めた 新しい版で HEAD の全 file に指摘 0 件

### Removed

- **破壊的:** `segment::compress_residual` (識別子を持たない加算の残差を書く 0.2.0-beta.2 の形式) を削除した 0.2.0-beta.2 以降 crate 内の呼び出しは試験だけで、識別子の無い残差を新たに書く経路を残さないため 残差は `compress_residual_xor` で書く

### Fixed
- 集計 (`aggregate` / `downsample` / `QueryBuilder::execute` の Scalar と Aggregates) が保存された NaN から計算した結果は正準の NaN `0x7ff8000000000000` を返す (ALICE-Law G4) これまでは保存された NaN の符号と payload がそのまま混ざった NaN (例 `0xfff81bd5a0000000`) を返していた `get` / `scan` は保存した値の bit をそのまま返す 試験 `tests/float_edge_golden.rs` に 4 件 (NaN の集計 / downsample / lossless の flush 後、1 が吸収される打ち消しの順序) conformance の G4 を pass に
- 既定の lossy (`FitConfig::lossless = false`) の flush で、保存した NaN / ±inf (と同じ segment の -0.0) が model の近似値に置き換わっていた (例: key 5 の NaN が 5.0 で読み戻る) 非有限の値を含む buffer は FitConfig によらず厳密に保存する (固定した XOR 残差、識別子つき) その buffer は lossless と同じく key の間隔が崩れる所で segment を切り、lossy の buffer に最初の非有限の値が来た時に間隔が揃っていなければ、それまでを有限値だけの lossy segment として先に閉じる `get` / `scan` は保存した値の bit をそのまま返す (payload 付きの NaN も) 残差の無い (lossy の) segment の点と model だけの読み (`query_point_model_only` / `query_range_model_only` / `generate_all`、`ResidualPolicy::ModelOnly`) は model の評価 (計算した値) なので NaN を正準の `0x7fc00000` にする (正準の NaN は bit で書く、std は `f32::NAN` / `f64::NAN` の bit を固定しない) (残差は正準化前の model 値に対して計算・適用するので、既存の lossless segment の読みは変わらない) 費用と制限: 非有限の値を含む segment は lossy 圧縮されない (残差を持つ) 試験 `tests/nonfinite_exact.rs` (5 件、両方の reader)
- `scripts/run_tests.sh` は `Doc-tests <crate>` の結果を最後に走った試験 binary に足していた (`verdict_record_semantics` が 3 件のところ 12 件と数えられ、`=3` の下限が doctest の件数で満たされていた) doctest は独自の節として数えて要約に `N doctests` と出し、節の見出しは cargo の字下げの後でだけ認める (試験の出力に出た `Running` で数え先が移らない) 試験は binary 複数 + doctest の log を再生し、旧版で red になることを確かめた
- 残差と法則の記録の識別子の不一致 (`ResidualMismatch` / `ResidualStatus::Mismatch` / `LawStoreError::LawRecordMismatch`) は、`running` に読みが実際に比べた識別子 (`alice_det_math::SEMANTICS_ID`) を入れる これまでは印に使う `alice-zip` の `law::SEMANTICS_ID` を入れていたので、両者が別の値になる `alice-zip` で build すると `stored` と `running` が同じ値の不一致を報告した 印 (`SEMANTICS_ID`) が `alice_det_math::check_semantics` の受け取る識別子であることは compile 時の assert で固定し、違う `alice-zip` (自分の識別子を持つ 0.11) では build が止まる 試験 `a_mismatch_reports_the_identifier_the_read_compared_with` (`tests/legacy_segment_compat.rs`)

## [0.5.1] - 2026-10-15

### Added

- CI: release profile で bit 一致 golden を走らせる `release-profile.yml` (ubuntu / macOS、`scripts/run_tests.sh` で 0 件と未実行の file を fail、preflight の full にも同じ step) と、公開 API の符号付きゼロと桁落ちの結果を `to_bits()` で固定する `tests/float_edge_golden.rs` (15 件: `aggregate` / `downsample` の Sum・Avg・Variance・StdDev、flush 後の constant / linear model と lossless 残差の `get` / `scan`) を追加
- `alice-law-conformance.toml` を追加し、ALICE-Law の G1〜G14 を現在の main で 1 件ずつ判定した (pass は G12 のみ、G1 / G3 / G9 / G10 / G13 / G14 は unverified、ほかは fail-known で該当の file と行を記載) README / README_JP の ALICE-Law 節を同じ revision の生成器で再生成した
- CI (G11、検査器自身の試験): clippy の禁止一覧 (G1 の超越関数、G10 の wall clock / OS 乱数) に canary を追加 `scripts/clippy_canary.py` が repo の `clippy.toml` / `rust-toolchain.toml` / `[lints]` を写した使い捨て crate に `scripts/clippy_canary_calls.tsv` の 1 行ごとに禁止呼び出しを 1 つ置き (`f64::sin` / `f64::sin_cos` / `Instant::now` / `SystemTime::now` / `RandomState`)、行ごとにその path を名指す `disallowed_*` 診断がちょうど 1 件出ること・対照の関数に 1 件も出ないこと・各行の path が `clippy.toml` にあることを確かめる data file が 0 行、または `clippy.toml` が 0 件の時は「比較 0 件」で失敗する 試験 `scripts/test_clippy_canary.py` と共に clippy job と preflight で走る 3 file は ALICE-Analytics と byte 一致
- CI (G11): `scripts/run_tests.sh` の試験 `scripts/test_run_tests.py` (ALICE-Analytics と byte 一致) を docs-lint job (3 OS) と preflight に追加
- CI: clippy job に `python` feature の step (`cargo clippy --lib --features python -- -D warnings`) を追加 これまで python の module は lint されていなかった
- 試験 (G11): `tests/determinism_golden.rs` に `assert_golden` 自身の canary を 4 本追加 (外部で計算した SHA-256 を受け入れる対照、下限より短い出力で panic、何も hash しない出力で panic、異なる hash で panic) 記録済みの golden 値は変えていない
- CI: libm を直接依存に足した時に G1 の禁止 (clippy.toml の `libm::` path と `scripts/libm_ban_check.py`) が無ければ失敗する検査 `scripts/libm_dependency_guard.py` (試験 `scripts/test_libm_dependency_guard.py`) を docs-lint job (3 OS) と preflight に追加 (ALICE-DetMath と byte 一致)

### Changed
- 版を 0.5.1 にした (公開済みの alice-edge 0.3.0 などが要求する `^0.5` の範囲で、`analytics` feature の alice-analytics 0.6 を受ける) determinism golden で既存の pin が変わったのは 2 つだけ: 算術の識別子 (`135c407d…` → `85e08846…`) と `law_id and evaluate` (`f18a2a53…` → `703f17b1…`) 旧い 2 つの値は PREVIOUS の pin として残り、0.5.0 の識別子を渡すと bit 単位で同じ値を再現する (`law_id_under_the_previous_identifier_is_unchanged`) 他の digest (model selection / 読み戻し / 集計 / bloom) は不変 識別子が動いたのは alice-det-math 0.5.1 によるもので、公開済みの alice-db 0.5.0 も alice-det-math を `^0.5` で要求するので、新しく解決すれば 0.5.0 でも既に 0.5.1 と新しい識別子になる 符号違いのゼロに対する `Aggregation::Min`/`Max` の変更 (下の Fixed) は 呼び出し元に返す集計値だけに効き、保存される bit は変わらない (0.5.0 からの src の変更は `query_engine.rs` の 4 行と `storage_engine.rs` の試験用 lint の許可の削除のみ)
- CI: 公開する文面の検査の pin を ALICE-Law 5de698e (checker sha256 4b3d9542…) に進めた 新しい版で HEAD の全 file に指摘 0 件

- CI: `scripts/clippy_canary.py` の説明を直した canary は repository の clippy job の flag に依らず決まった flag (`-W clippy::pedantic -D warnings`) で試す (以前は「CI の clippy job と同じ flag」と書いていたが、job によって flag は違う) 動作は変わらない (ALICE-Analytics / ALICE-DB / ALICE-Zip / ALICE-Edge と byte 一致)
- ALICE-Law 適合 manifest の G2 と G4 の note を現状に合わせた Aggregation::Min / Max と group の min / max は alice-det-math の符号の決まった関数に置き換わり、`tests/minmax_signed_zero_arch.rs` で固定されている (G2 は禁止と manifest が無いので fail-known のまま、G4 は桁落ちの順序と branch の保護が残るので fail-known のまま) G4 の「float_edge_golden が必須の一覧に無い」は既に解消していたので消した
- **Behavior change:** `alice-det-math` の要求を 0.5 から 0.5.1 に上げ、Cargo.lock も 0.5.1 に進めた `analytics` feature の `alice-analytics` 0.6 が `alice-det-math` 0.5.1 を要求するので、DB の要求が 0.5 のままだと 0.6 を解決した時だけ 0.5.1 が入り、`tests/determinism_golden.rs` の識別子と法則の pin が落ちていた 0.5.1 で算術の識別子 `alice_db::SEMANTICS_ID` が `135c407d…` から `85e08846…` に変わる 旧い `135c407d…` は `alice_det_math::PREVIOUS_SEMANTICS_IDS` に残るので、それで stamp した残差と、その識別子で索引した法則は読み取り時に引き続き受理される `law_id` は識別子を混ぜるので値が変わる 旧い識別子を渡すと以前の `law_id` がそのまま出るので、変わったのは識別子だけで、法則の評価値と保存される bit (model selection / 読み戻し / 集計 / bloom) は変わらない
- `analytics` feature の `alice-analytics` の要求を `0.5` から `>=0.5, <0.7` に広げ、0.6 も解決できるようにした
- CI: 公開する文面の検査の pin を ALICE-Law f9b9000 (checker sha256 84226ec7…) に進めた 新しい版で HEAD の全 file に指摘 0 件
- CI: clippy の禁止一覧に `f32::sin_cos` / `f64::sin_cos` を足した (ALICE-Law G1 の一覧と揃える)
- CI: debug profile の test step 3 本 (既定 / 全 native feature / 既定 feature 無し) の `scripts/run_tests.sh` の必須一覧に `float_edge_golden` を足し、この file が 1 件も走らない時に失敗させる (preflight も同じ)
- `alice-law-conformance.toml`: G1 を pass に (python を含む全 feature を lint、canary あり、libm への直接依存なし) G11 は fail-known のまま証跡と変異と note を更新 (未達は共有の写しの照合) README の ALICE-Law 節を作り直した

### Fixed

- **Behavior change:** `Aggregation::Min`/`Max` (`aggregate` の plain fold と `group_by` / `downsample` の bucket の両方、`query_engine.rs`) を `f64::min`/`max` / `f32::min`/`max` から `alice_det_math::{min64, max64, min, max}` に変えた 等しい符号違いのゼロ (`-0.0` と `+0.0`) に対する `f64::min`/`max` の結果は規定されておらず、aarch64 は IEEE の minimum / maximum (順序に依らず `min` は `-0`、`max` は `+0`)、x86_64 は同値のとき第 2 引数を返すので、同じ多重集合でも arch と行の順序で符号が変わっていた 実測 (16 入力 × plain / grouped × Min / Max の 64 出力を aarch64 / x86_64 × debug / release で `to_bits` 比較): 修正前は混在ゼロの 4 入力の 8 出力だけが x86_64 で aarch64 と異なり、修正後は 4 構成の全出力が一致 変わったのは x86_64 のその 8 出力だけで、aarch64 と混在ゼロ以外の入力 (非ゼロ / +0 のみ / -0 のみ / NaN / 大きな値 / 0..99) は不変 試験 `tests/minmax_signed_zero_arch.rs` (両方の順序、plain と grouped) 4 箇所のどれか 1 つを `f64::min`/`max` / `f32::min`/`max` に戻すと x86_64 で red (aarch64 は IEEE どおりなので green)
- `alice-det-math` の要求を 0.5.1 に上げた (符号を決めた `min`/`max`/`min64`/`max64` は 0.5.1 で追加) 0.5.1 で `SEMANTICS_ID` が `135c407d…` から `85e08846…` に変わるので、`tests/determinism_golden.rs` の識別子と law_id の 2 つの pin を記録し直した 動いたのは識別子の入力だけで、0.5.0 の識別子 (0.5.1 の `PREVIOUS_SEMANTICS_IDS` にある) を渡すと law_id と evaluate の digest は以前の `f18a2a53…` のまま (同じ file の試験で固定)、他の digest は不変 0.5.0 の識別子で書いた残差は `check_semantics` が受理するので読み戻せる

## [0.5.0] - 2026-10-14

### Changed

- **Behavior change:** `residual_is_applicable` / 残差の読み取り経路が、pin された識別子を `alice_core::law::SEMANTICS_ID` との生の等値比較でなく `alice_det_math::check_semantics(..).is_reproducible()` で判定するようにした 生の等値比較は `alice-det-math` が関数を追加するだけで識別子が動いた時 (bit は 1 つも変わっていなくても) 既存の segment を拒否してしまう `check_semantics` は現在の識別子 (`Verified`) に加え、同じ bit を保証する過去の識別子 (`VerifiedPrevious`、`alice_det_math::PREVIOUS_SEMANTICS_IDS`) も受理し、未知の識別子だけを拒否する (`Mismatch`) oracle: 0.5.0 自身の識別子 (frozen な byte 列、依存から読まない) で stamp した残差が読み戻ること、未知の識別子は拒否されること 変異試験: 2 箇所の判定をそれぞれ生の等値比較に戻すと red (この crate が現在依存する `alice-det-math` 0.5.0 だけでは識別子が動いていないため見えない差で、識別子が動いた `alice-det-math` 0.5.1 への path 上書きで local 確認、commit には含めない)
- `Cargo.toml` の `[package]` に `include` を追加し、publish される `.crate` を src / tests / benches / examples / bindings / README / CHANGELOG / LICENSE 類に限定した (CI 設定・scripts・fuzz・docs は含まない) CI の package integrity job が、`.crate` の file 一覧にこの crate のものでない path が無いこと (一覧が 0 件なら fail) と、展開した `.crate` が `--all-targets` で build でき doctest も通ることを検査する

### Added

- `AliceDB::law_pointers_for` / `AliceDB::reindex_laws_under_current_semantics`: `law_pointers_by_id` は書かれた時の識別子だけに答えるため、`alice-det-math` の識別子が (bit を変えずに) 動いた後は古い識別子を知らない限り以前の law に届かなかった `law_pointers_for` は現在の識別子に加え `alice_det_math::PREVIOUS_SEMANTICS_IDS` (`check_semantics` が受理する過去の識別子) を順に試し、どちらで見つかったかを返す (`LawPointersFoundUnder::Current` / `Previous`、書き込みは一切しない) `reindex_laws_under_current_semantics` は、現在の識別子または `PREVIOUS_SEMANTICS_IDS` のいずれかで見つかった law にだけ現在の識別子のポインタを追加で書く (冪等、古いポインタは消さない、どちらでも見つからない — `SemanticsCheck::Mismatch` に相当する — law は触らない) oracle: 現在の識別子で見つかる・どの識別子にも無い・reindex が追加のみで冪等・全 name の全 version を reindex する・**受理されない識別子の law は reindex しても放置される**、の 5 本 変異試験: fallback の削除、reindex の冪等性チェックの削除、reindex が最新 version しか辿らない、**reindex が受理判定を飛ばして無条件に現在の識別子を書く**、の 4 本が red 「見つかる previous」自体は、この crate が現在依存する `alice-det-math` 0.5.0 の `PREVIOUS_SEMANTICS_IDS` がまだ空のため commit した test では見えない差になる (`check_semantics` の時と同じ構造) ため、識別子が動いた `alice-det-math` 0.5.1 への path 上書きで local 確認 (found/miss/reindex/reindex の受理判定の 4 oracle green、fallback 変異・受理判定を飛ばす変異とも red)、commit には含めない `reindex_laws_under_current_semantics` は内部で `pub(crate) reindex_with(semantics_id, accepted)` に委譲するよう分割した (`accepted` を明示引数にしたもの) `PREVIOUS_SEMANTICS_IDS` が空の間は「正しく絞り込んでいる」と「常に 0 を返すだけの stub」が公開 API からは見分けられないため、`reindex_with` に注入した identifier で「受理された識別子の law は追加される」「受理されない識別子の law は追加されない」「冪等」の 3 本を crate 内部 test として追加、両方向の変異 (常に skip する・絞り込みを飛ばして常に追加する) が red
- Python: `open` / `AliceDB(...)` に `lossless=False` 引数を追加 既定は従来どおり lossy (gap のある系列は等間隔 grid で読み戻され、flush 後に gap の点の値が変わる)、`lossless=True` で残差を保存し増加するタイムスタンプを厳密に読み戻す (Rust の `FitConfig { lossless: true }` と同じ約束) README / README_JP の Python 節に lossy 既定と lossless の例を明記

### Fixed

- Python: crates.io から取得した利用者の環境で `python` feature が compile 不能 (E0308、`PyArray2::from_owned_array` の引数型の不一致) だった 直接依存の `ndarray = "0.16"` と、`numpy 0.29` が新たな解決で選ぶ `ndarray 0.17` の 2 版が並び、型が別物になる repo の Cargo.lock が 0.16 に揃えていたため CI と preflight では見えなかった 直接依存の `ndarray` (と同名の optional feature) を外し、`numpy::ndarray` の再公開を使う形にして、numpy が使う版と常に一致させる security-audit の package-integrity job が、展開した .crate から Cargo.lock を外して `cargo check --lib --all-features` を走らせる
- CI (`clippy.toml`、G10): `std::time::Instant::elapsed` / `std::time::SystemTime::elapsed` を disallowed-methods に追加 どちらも `::now()` と同じく wall clock を読むが、どちらの層も検査していなかった (`.elapsed(` の字面検査は `elapsed` という名の値全般を拾い過ぎるため、型解決する clippy 側だけに持たせた) `duration_since` はすでに保持した値どうしの引き算で clock を読まないため対象外 repository 内の唯一の呼び出し (`storage_engine.rs` の `#[cfg(test)]` module の中、計測した `Duration` を使わない dead code) は削除した canary: clippy.toml にこの 2 entry があることを確認する test canary の一覧 `scripts/clippy_canary_calls.tsv` に 2 行 (保持した値の `.elapsed()`) を足して 7 行にした (ALICE-Analytics と byte 一致) ALICE-Law 適合 manifest の G10 を pass にした
- `transaction::LockManager::detect_deadlock()`: wait-for グラフと DFS の循環検出用集合 (`wait_for` / `visited` / `in_stack`、`dfs_cycle` の `graph` 引数含む) を `HashMap`/`HashSet` から `BTreeMap`/`BTreeSet` に変更 (G13) `HashMap`/`HashSet` の反復順序は `RandomState` によりインスタンスごとに乱択されるため、同じ論理内容の wait-for グラフでも `detect_deadlock()` を呼ぶたびに DFS の開始ノード・探索順が変わりえた (`detect_deadlock()` は呼び出しのたびに内部で新しい `HashMap` を構築する) 複数の transaction が同じロック群を取り合う待ち状態 (2 組以上の独立した循環を含む、実世界では同じロック構成に別々の順序で到達しうる) では、どの循環を「検出された循環」として返すかが呼び出しごとに変わり、同じロック状態に対して異なる deadlock victim を非決定的に報告しうる bug だった `TxnId` (`u64` の type alias) は元から全順序を持つため、`BTreeMap`/`BTreeSet` への置換は追加の trait 実装を要さない `transaction.rs` の残り 7 field (`MvccStore::versions`/`txn_status`/`txn_start_ts`/`txn_write_set`、`LockManager::locks`/`held_locks`/`LockEntry::holders`) は全 call site を確認した上で point lookup のみ、または反復しても戻り値が反復順に依存しない (独立した変更 + 可換な集合構築 + 反復順不問の件数) ことを確認し、据え置いた (分類根拠: `docs/g13-hashmap-manifest.md`) 新規 oracle 2 本: 同一ロック状態に対する `detect_deadlock()` の 200 回繰り返し呼び出しが毎回同じ結果を返すこと、および同じ論理状態を異なる挿入順序で構築しても同じ結果を返すこと (変異: 型を `HashMap`/`HashSet` に戻すと 5/5 回で異なる循環を観測、同一プロセス内でも再現)
- G13 (非決定な反復順の排除): `StorageEngine::stats()` / Python `Stats` の `model_distribution` (segment ごとの model 種別の集計) が `HashMap<String, usize>` だったため、同じ集計結果でも `Debug` 出力の反復順が実行ごとに変わり得た (同じ model 種別の多重集合を異なる flush 順で作った 2 つの `StorageEngine` で実測、`{"Constant": 2, "Linear": 1}` と `{"Linear": 1, "Constant": 2}` のように順序が食い違った) `BTreeMap<String, usize>` に変更し、キーの順序で一意に定まるようにした 公開 API は `StorageStats::model_distribution` (Rust) と `Stats.model_distribution` (Python、pyo3 の `BTreeMap` 変換は `HashMap` と同じ形で効く) の 2 箇所、両方揃えて変更 値そのものは変わらない `replication.rs` の `RaftNode::next_index` / `match_index` (`HashMap<NodeId, LogIndex>`) は全呼び出しが `insert` / `get` / `get_mut` / `clone` のみで直接の反復が無く、クラスタのメンバー管理専用のため対象外 (全 5 箇所を確認)
- CI (`scripts/entropy_guard.py`、G10): 2 つの穴を修正 (1) `RandomState::new` だけを検査しており `RandomState::default()` (同じ OS 乱数シードのハッシャーに届く) を素通りさせていた `RandomState` の裸の語を検査する形に変更 (clippy 側の `disallowed-methods` は `Default::default` をトレイトメソッドとして実装型だけでは指せないため、この script が字面で検査し、clippy の `disallowed-types` (`RandomState` 型そのものを禁止) が型解決した上で検査する 2 層構成にした) (2) `#[cfg(test)]` module 判定のブレース深さ計算が文字列しか除去しておらず、`'{'` のような char literal がブレース深さを狂わせて test module 判定が早期終了し、直後の production code を見逃す経路があった 文字列・raw string・char literal・ブロックコメントを 1 pass で正しく除去する sanitizer に置換 (3) `<std::time::Instant>::now()` のような qualified-self 呼び出しが `Instant::now` のパターンに `>` を挟むため素通りしていた (`as Trait` を挟む形も含む) 型名と `::method` の間に `>` / `as Trait` を許容するよう正規表現を拡張 canary test 6 本追加 (元の script に対して赤を確認) 比較件数は変わらず 23 file / 10 pattern / 0 problems
- CI: ci / fuzz / security-audit の concurrency を main では commit ごとの group で打ち切らない形に変更 (branch と PR だけ新しい run で打ち切る) ci.yml には top-level の concurrency を追加 `scripts/workflow_concurrency.py` + `scripts/test_workflow_concurrency.py` (他 repo と byte 一致) が main の打ち切りと group の欠落を検査する (比較 0 件で fail、docs-lint job と preflight に配線)
- Python: README / README_JP の quick-start 例を修正 `put_batch` は `(timestamps, values)` の長さの等しい 2 列を取る (pair の list を渡すと `ValueError`/`TypeError` だった) 例は `python/tests/test_readme.py` が verbatim に実行する source of truth
- Python: wheel の version を Cargo.toml に一元化 pyproject を `version = "0.1.0"` から `dynamic = ["version"]` に変え、maturin が Cargo.toml の version を使う wheel が 0.1.0 固定で crate 0.4.0 と drift していた python CI が `alice_db.__version__ == Cargo.toml の version` を検査する
- Python: wheel の license metadata を crate の宣言に一致させた pyproject が `MIT` を宣言していたが、crate は `AGPL-3.0-or-later OR LicenseRef-Commercial` (Cargo.toml / LICENSE-AGPL / LICENSE-COMMERCIAL.md / README) pyproject を SPDX 式 + `license-files` にし、MIT classifier を除去 `scripts/check_py_license.py` が pyproject と Cargo.toml の license 一致を検査する (0 件比較で fail)
- CI: build した library と実行形式を tracked にしない検査 (`scripts/tracked_binaries.py`) を追加 例外は `.github/tracked-binaries.txt` に理由付きで書いた path だけ、読めない file と走査 0 件でも失敗する CI と preflight で走る
- CI / `scripts/preflight.sh`: python feature を build + test する `python` job を追加 (`maturin build` → wheel install → import smoke → `pytest python/tests`、collected 0 件で fail) これまで CI は python を一切 build しておらず (`cargo hack` も `--exclude-features python`)、binding と README の例が壊れても気付けなかった
- CI: `scripts/changed_paths.py sync` が各 workflow の Gate job の `needs` に他の全 job が入っているかを確かめる (入れ忘れた job は Gate を通らずに失敗や skip が見えなくなるため) `changed_paths.py` と試験は ALICE-DetMath と byte 一致
- CI / `clippy.toml`: 決定論 gate (G10) として `std::time::Instant::now` / `std::time::SystemTime::now` / `std::collections::hash_map::RandomState::new` を disallowed-methods に追加 segment のモデルと残差は書いた機械と別の機械で読み戻されるため、production code が wall clock や OS randomness を読む経路を持つと再現できない `scripts/entropy_guard.py` (試験 `scripts/test_entropy_guard.py`) が同じ対象を compile 無しで静的に検査する (許容する呼び出し箇所 0、下記 `created_at` の破壊的変更で解消) docs-lint job (3 OS) と preflight で走る
- **破壊的:** `segment.rs` の `created_at` メタデータが `SystemTime::now()` を直接呼んでいたため、G10 の「production code は wall clock を読まない」を満たせなかった (検証で発覚: 同じ引数で数 ms 間を空けて呼んだ 2 回の `DataSegment::new` が異なる byte 列にシリアライズされていた) `DataSegment::new` に 7 番目の引数 `created_at: u64` を追加し、呼び出し元が明示的に渡す値をそのまま記録する (crate は一切 wall clock を読まない) `StorageConfig` に `created_at: u64` (既定 0) を追加し、`MemTable::with_config_and_created_at` (新規、既存の `new` / `with_config` は `created_at: 0` を内部既定にして署名は変更なし) 経由で `StorageEngine` から渡す 消費者調査: workspace 内で `DataSegment::new` を直接呼ぶ crate は無し (`cargo semver-checks` が実際に検出する破壊は `constructible_struct_adds_field` (`StorageConfig::created_at`) と `method_parameter_count_changed` (`DataSegment::new` 6→7) の 2 件、0.3→0.4 の時と同型) oracle 3 本を新規追加: (1) `segment.rs` — 同一引数 (同一 `created_at` 含む、呼び出し間に 5ms の間隔を空ける) の 2 回の `DataSegment::new` が `to_bytes()` で byte 一致 (変異: `created_at` を実クロック読みに戻すと red、間隔を空けない版では偶然同一 ms に収まり green になりうることを確認した上で間隔を追加) (2) `storage_engine.rs` — `StorageConfig::created_at` が `StorageEngine` → `MemTable` → `DataSegment::new` の配線を経て flush 済み segment の metadata に実際に届くこと (変異: 配線の呼び出しを `0` 固定に戻すと red、固定前は他の 419 test 全部 green のまま見逃していた) (3) 同じ `created_at` を与えた 2 つの独立した engine が byte 一致する segment file を書くこと 版を 0.4.0 から 0.5.0 に
- CI (`fuzz.yml` / `security-audit.yml`): `on.push` から `paths` を外し、`changes` job (`scripts/changed_paths.py decide`) が実行の要否を決める GitHub の `paths` は branch を作る push では repo に新しい commit だけを比べるので、既に別の ref にある SHA で `ci/**` branch を作ると差分が空になり fuzz と security の job が走らなかった 新しい branch と force push は `merge-base(origin/main)..after`、それ以外の push は `before..after` (main も同じ)、pull request は `base...head` で比べ、schedule と手動実行は常に走る git の失敗・merge-base が無い・`before` に届かない・新しい branch で差分が空のときは走らせる (理由を log と job summary に出す) pattern は `.github/paths/fuzz.txt` と `.github/paths/security-audit.txt` に置き、`pull_request` の `paths` と一致することを `changed_paths.py sync` が検査する 最後の `Gate` job は run=true なら対象の job の成功を、run=false ならその skip を求める (`security-audit.yml` の informational な coverage / semver-checks は `Gate` の対象外で、run=false の時は skip する) `Gate` は required status check にしない (`pull_request` には `paths` が残るので、対象の path に触れない pull request では run が無い) 判定器の試験 `scripts/test_changed_paths.py` と `sync` は docs-lint job (3 OS) と `scripts/preflight.sh` で走る `scripts/changed_paths.py` と `scripts/test_changed_paths.py` は ALICE-LOL と同一
- CI: 公開する文面の検査 (`Public text` workflow、`scripts/public_text.sh`) を追加 ALICE-Law の `scripts/public_text_check.py` を `.github/public-text-pin` の revision で取得し、SHA-256 が一致した時だけ使って、HEAD の全 tracked file (path と本文) と送った commit (message・追加行・追加 path) を読む 何も比較できなければ失敗する preflight でも走る `.public-text.toml` は文書の linter とその試験が意図して持つ語を、種類ごとの行数で許可する (行数が変われば失敗)

### Security

- CI (`ci.yml` / `fuzz.yml` / `security-audit.yml`): workflow 全体の token を `permissions: contents: read` にし、書き込みが要る job だけ理由の comment を付けて上げる (actionlint の reviewdog が check run を書くので `checks: write`) 外部の action 54 箇所をすべて 40 桁の commit SHA に固定し、tag か branch を comment に残す tag と branch は別の code に付け替えられるので、参照を内容に固定する branch 参照 (`dtolnay/rust-toolchain@stable` / `@1.87` / `@nightly`、`taiki-e/install-action@cargo-hack` / `@cargo-llvm-cov`) は action.yml の既定値が toolchain と tool を決めるので、SHA に固定しても入る版は変わらない
- 全 `actions/checkout` に `persist-credentials: false` を付けた どの job も push しないので、token を `.git/config` に残さない `pull_request_target` / `workflow_run` の trigger は使っていない
- `scripts/workflow_pin_check.py`: 全 workflow に top-level の `permissions:` があり、外部の `uses:` がすべて 40 桁の SHA + comment で、job の `write` に理由の comment があり、`actions/checkout` が `persist-credentials: false` で、`pull_request_target` / `workflow_run` の trigger が無いことを検査する `uses` の key を含む行はすべて照合し (引用符・式・空白を含む値も unpinned として fail、認識できない行と照合件数の不一致も fail)、`permissions:` は top-level と job のどちらでも inline の write (`write-all` / `{contents: write}`) と理由の comment の無い block の `write` を fail にする `.github/actions/**/action.yml` の `uses:` と checkout も同じ規則で見る workflow が 0 件・`uses:` が 0 件なら fail `--online` は `gh api` で各 SHA が comment の名指す commit か (tag はその SHA に解決する、branch はその SHA を含む) を確かめ、比較 0 件で fail する CI では Linux の docs-lint job で 1 回、変更された pin だけを見る (下の項目) 試験 `scripts/test_workflow_pin_check.py` (28 本、online は偽の API、base の解決は一時の git repository)
- `scripts/workflow_pin_update.py`: pin を手で更新する script `# vX.Y.Z` は major tag `vX` の現在の commit とそれを指す最新の release tag に、`(branch, ...)` は branch の先頭に動かす 新しい major には動かさない 既定は diff の表示だけで、`--write` で書き換え、commit はしない 手順は `CONTRIBUTING.md` 試験 `scripts/test_workflow_pin_update.py` (6 本) 以上の試験は docs-lint job (3 OS) と `scripts/preflight.sh` で走る
- CI の online の pin 検査を、その変更で追加・変更された pin だけにした (`--online --base auto`、push 前の commit か pull request の base と比べる、変更が無ければ「no changed pins」で通る) 変わっていない pin が今も tag と一致するかは action 側の repository に依存し、この repo を変えなくても結果が変わる 実際に `taiki-e/install-action` が tool ごとの branch (`cargo-hack` / `cargo-llvm-cov`) を release のたびに作り直し、固定した commit が branch から外れて main の docs-lint が落ちた 全 pin の online 検査は週 1 回の `pin-audit.yml` (読み取りの token、失敗は更新の合図) と `workflow_pin_update.py` で行う `scripts/preflight.sh` も origin/main からの差分だけを見る
- `taiki-e/install-action` は作り直される branch でなく release tag (`v2.87.27`) に固定し、`tool:` で `cargo-hack` / `cargo-llvm-cov` を指定する branch に固定するのは前にしか進まない branch (`dtolnay/rust-toolchain`) だけにする
- `workflow_pin_update.py` は固定した commit を含まなくなった branch を更新せず報告する 更新の後に全 pin の online 検査を実行する
- `--base auto` は event から得た commit (pull request の base / push 前の commit) が検査中の commit 自身の時も、merge base の時と同じく親の commit と比べる (自分自身と比べると変更された pin が 0 件に見える)

## [0.4.0] - 2026-10-12

### Changed

- **Behavior change:** `alice-det-math` を 0.4 から 0.5 に、`alice-zip` (`alice_core`) を 0.9 から 0.10 に、`analytics` feature の `alice-analytics` を 0.4 から 0.5 に上げた 算術の識別子 `alice_db::SEMANTICS_ID` が `d2209b30…` から `135c407d…` に変わる 法則の評価値は変わらない (determinism golden の `law_id and evaluate` は旧識別子で計算すると記録済の `94bf9625…` のまま、変わったのは識別子だけ) `law_id` は識別子を混ぜるので値が変わる
- **破壊的:** `segment::apply_residual` を crate 内部 (`pub(crate)`) にした 補正の語が無い index で値をそのまま返しており、外から呼ぶと model の値が原信号の顔で返る経路だった (crate 内の読み出しは長さを先に検査するので通らない) 内部では `Option<f32>` を返し、語の無い index は読み出しが `ResidualError::Corrupt` にする 補正した値は `query_point` / `query_range` で読む ALICE-* の main に呼び出し元は無い
  - `cargo semver-checks` (0.3.0 との比較): 破壊的 2 件 = `function_missing` (`segment::apply_residual`) と `constructible_struct_adds_field` (`StorageConfig::residual_policy`) 一方で `DataSegment` / `SegmentView` の `query_point` / `query_range` / `query_range_simd` の戻り値が `Option` / `Vec` から `Result` に変わったことは検出しない (method の戻り値の型の変化はこの版の lint に無い) が破壊的
- **Behavior change (破壊的):** 点ごとの補正に復号できない lossless 残差 (LZMA として解けない payload、途中で切れた識別子、未知の形式、点数に足りない長さ) を、読み出しが黙って model の値で返すのをやめ、`ResidualError::Corrupt` (`ResidualCorrupt`: segment id・key 範囲・必要な byte 数・実際の byte 数・store) で断る 0.3 までは識別子が一致していても中身が壊れていれば model の値を原信号として返していた (例: 正しい識別子 + LZMA でない 7 byte の残差で `query_point` が `Ok(Some(model))`) `residual_status()` は `ResidualStatus::Corrupt` を返し、`ResidualPolicy::ModelOnly` と `*_model_only` は従来どおり model を返す 壊れた残差はどの版でも取り戻せないので、segment を backup から戻す 壊れた blob 9 形のどれでも panic しない 0.3.0 より前の形式の正常な残差は従来どおり読める
- **Behavior change (破壊的):** 別の算術の識別子を持つ lossless 残差を、読み出しが黙って model の値で返すのをやめ、型付きの error で断る alice-db 0.3 (alice-det-math 0.4) が書いた lossless の store を本版で読むと、その segment の key は `get` / `scan` / `query` が `io::ErrorKind::InvalidData` で失敗し、内側の `alice_db::ResidualMismatch` (segment id・key 範囲・store の directory・保存された識別子・実行中の識別子) を `get_ref().downcast_ref()` で取り出せる 範囲読みは全件か error のどちらかで、件数の少ない結果や一部だけ補正した結果は返さない (より新しい書き込みが全部の key を上書きした古い segment は読みを失敗させない) 0.3 までは識別子の不一致を model の値で返しており、原信号でない値を原信号として返していた (64 点の lossless 系列で 64 点とも原信号と異なり、最大 2,060 万 ulp)
  - model の近似を明示して読む: `StorageConfig::residual_policy = ResidualPolicy::ModelOnly` (既定は `ResidualPolicy::Refuse`)、segment では `query_point_model_only` / `query_range_model_only`
  - 読まずに状態を見る: `DataSegment::residual_status()` / `SegmentView::residual_status()` (`ResidualStatus::{Exact, NoResidual, Unpinned, Mismatch}`)
  - 厳密な値を取り戻すにはその残差の算術が要る: alice-db 0.3 で読んで本版で書き直す 本版だけでは model の値しか得られない
  - 識別子を持たない 0.3.0 より前の残差は従来どおり適用する exact series (`AliceDB::series`) は残差を使わないので影響しない
  - API: `DataSegment` / `SegmentView` の `query_point` は `Result<Option<f32>, ResidualError>`、`query_range` (と `query_range_simd`) は `Result<Vec<(i64, f32)>, ResidualError>` を返す (`ResidualError::{Mismatch(ResidualMismatch), Corrupt(ResidualCorrupt)}`、`ResidualError` / `ResidualMismatch` / `ResidualCorrupt` / `ResidualStatus` / `ResidualPolicy` はすべて `#[non_exhaustive]`) `StorageConfig` に `residual_policy` を足した (struct literal で全 field を書く呼び出しは `..StorageConfig::default()` が要る)

## [0.3.0] - 2026-10-11

### Fixed

- 時系列 store の範囲読み (`AliceDB::scan` / `query_range`、`DataSegment` と zero-copy の `SegmentView`、多項式の SIMD 経路を含む) が終わらないことがあった f64 の時刻を格子の step ずつ足しており、2^53 を超える key で step が ulp の半分より小さいと時刻が進まず、結果を伸ばし続けて メモリを使い切った (2^62 付近の連続した key で再現) 格子点を整数の index で歩き、時刻は `query_start + ⌊k·range/(n−1)⌋` を 128 bit 整数で正確に出す 1 回の読みが segment ごとに返す点は `point_count` 以下 値を求める sample 位置は従来と同じ f64 演算になる形で出すので、整数の step で 2^53 未満の key では読み戻す bit は変わらない (`tests/determinism_golden.rs` の digest は不変)
- segment の時刻の差 (`end − start`、`t − start`) を i64 で引いていた 幅が `i64::MAX` を超える segment (正負をまたぐ key) で debug build は panic、release build は折り返して別の位置を読んでいた i128 で引く 退化した segment (`point_count ≤ 1` で幅が正) は従来すべての整数時刻を返していたが、先頭の 1 点だけを返す
- `AliceDB::get` が memtable を読まず、flush 前に書いた値が `None` になっていた 複数の segment が key を覆う時は index が最初に返した segment が答えていた memtable (同じ key は最後の書き込み) を先に読み、次に segment を新しい順 (id の大きい順) に読む `scan` も memtable の点を含め、同じ key は memtable、次に新しい segment の値を残す
- **挙動の変更:** `QueryBuilder::group_by` / `downsample` の間隔が 0 か負の時、`execute` が panic していた (0 は除算、負は bucket 数の見積もり `((last − first) / interval + 1) as usize` が負になり、usize への変換で巨大な容量を確保しようとして capacity overflow) 結果を返す経路でも負の間隔の bucket は意味を持たない 間隔は正に限り、0 と負は点の有無によらず `io::ErrorKind::InvalidInput` で拒む 正の間隔でも key の幅が `i64::MAX` を超えると `last − first` が桁あふれしていた bucket 数は i128 で求め、点の数を上限にする (各 bucket は 1 点以上を持つ) `fuzz_query_parse` が見つけた入力 (間隔 −1) と、その間隔だけを 0 にした入力を `fuzz/regressions/fuzz_query_parse/` に置き、`tests/query_group_by_interval.rs` が 0 / −1 / −7 / `i64::MIN` の間隔 (flush 前・flush 後・空の store、`downsample` を含む) と ±2^62・i64 の両端の key を検査する memtable を読むようになった `scan` (上の項目) で、flush 前の点にもこの経路が届くようになっていた
- **挙動の変更:** `QueryBuilder::group_by` / `downsample` の bucket の label を 0 方向への切り捨て `(t / interval) * interval` で求めていた そのため bucket 0 は `2·interval − 1` 個の key を持ち (interval 10 で −9..=9)、負の key は 1 つ上の bucket に入っていた (−1 → 0、−11 → −10) floor の bucket にした: key `t` は `[k·interval, (k+1)·interval)` (`k = ⌊t / interval⌋`、i128 で計算) に入り、どの bucket も `interval` 個の key を持つ 例: interval 10 で −1 → −10、−10 → −10、−11 → −20、9 → 0、10 → 10 負の key を含む集計の結果が変わる 返す label は `max(bucket_start, i64::MIN)`: `i64::MIN` (−2^63) の bucket は 2^63 を割り切らない interval では `i64::MIN` より下から始まる (interval 7 で `i64::MIN − 6`、3 で −1、10 で −2、`i64::MAX` で −2^64 + 2) その最下位の bucket だけ label を `i64::MIN` にし、所属は floor のまま、label は狭義単調増加で重ならない bucket 数の見積もりも同じ floor の規則で求める 点の無い bucket は従来どおり返さない `tests/query_group_by_floor.rs` が bucket の幅 (interval 1 / 3 / 7 / 10、key −25..=25)、0 をまたぐ label、負の key の sum / avg / count / min / max、i64 の両端の所属と label を i128 の参照と比べる
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

- **破壊的変更 (`analytics` feature):** `alice-analytics` を 0.2 から 0.4 に上げた `analytics_bridge` の `flush_metrics_to_db` / `AnalyticsSink` が受け渡す `MetricPipeline` などは依存先の版ごとに別の型なので、`alice-analytics` 0.2 / 0.3 の型を作って渡している利用者は compile できなくなる (利用者も 0.4 に上げる) 0.3 以降は `alice-det-math` 0.4 を要求するので、依存グラフの det-math が 1 版 (0.4.0) になり、保存した Law の評価と分析側の推定値が同じ算術で計算される (従来は 0.3.2 と 0.4.0 が同居していた) `alice-analytics` は path + version の依存から公開版への依存に替え、CI は analytics ブリッジを実物に対して build する
- **破壊的変更 (`analytics` feature):** `flush_metrics_to_db` / `AnalyticsSink` は metric を時系列 store (`AliceDB::put_batch`) に書かず、exact series `"alice-analytics/metrics"` (`METRICS_SERIES`) の `metric_key` に書く 関数の signature は変えていないが、metric を `AliceDB::get` / `scan` で読んでいた利用者は何も読めなくなる `read_metric` / `scan_metric` / `AnalyticsSink::read` / `scan` で読む ブリッジに渡す値 (counter / gauge / cardinality / p50 / p90 / p99 の f32) は従来どおりで、0.3.0-beta.2 と同じ event 列に対し `flush_metrics_to_db` に渡る 720 値が全て一致することは確認済 (HLL と DDSketch の `ln64` / `exp64` も det-math 0.3.2 と 0.4.0 で 400 万入力ずつ一致) `alice-analytics` 0.4 / `alice-zip` 0.9 の組でも同じ 720 値が一致し、`read_metric` で読み戻す 720 値は書いた値と bit 単位で一致する **0.3.0-beta.x が時系列 store に書いた metric は移行しない** 時系列 store に残り、`AliceDB::get` / `scan` で `metric_key` を引けば読めるが、上の Fixed の理由で別の key の値が混ざる (正確な値は取り戻せない)
- **破壊的変更 (lossless mode):** `FitConfig { lossless: true }` の `put` / `put_batch` は、既に書いた key 以下の key を `io::ErrorKind::InvalidInput` の `io::Error` (中身は `LosslessKeyOrderError { key, last }`) で拒む batch は全体を先に検査し、拒んだ呼び出しは何も書かない 開き直し後は保存済みの最大 key から続く 従来は受け付けて別の key の値を返していた 上書きや順不同の key が要る値は exact series に置く 既定の非可逆 mode は従来どおりどの順の key も受け付ける
- **破壊的変更 (`alice-zip` の型の再 export):** `alice-zip` を 0.7 から 0.9 に上げた `alice_db::law_store` は `alice-zip` の `SignalLaw` / `SignalLawParts` / `Verdict` / `LawError` / `OracleCase` / `OracleOutcome` / `Provenance` / `ResidualStats` / `ValidRange` / `IngestPolicy` を再 export しており、これらは依存先の版ごとに別の型なので 0.9 の型になる 利用者が自分の `alice-zip` 0.7 / 0.8 で作った `SignalLaw` などは `put_law` に渡せなくなる (利用者も `alice-zip` 0.9 に上げる) crate root が再 export する `SEMANTICS_ID` の値は変わらない 公開済の 0.3.0-beta.2 は `alice-zip` ^0.7 を要求していたため、それより新しい `alice-zip` を使う crate と組むと `alice-zip` が 2 版 link されていた 保存済みの記録に関わる値は変わらない: `tests/determinism_golden.rs` の `SEMANTICS_ID`・model 選択・`law_id` の digest は 0.9 でも同じ値で通る
- 依存を `alice-crypto` 0.1 → 0.4 に上げた `alice-crypto` は path + version (`../ALICE-Crypto`, `"0.1"`) から crates.io の公開版への依存に替えた (隣の checkout は既に 0.3.0 で、要求 `"0.1"` を満たさず build できなかった) 本 crate が使う `seal` / `open` / `derive_key` / `hash` / `Key` / `CipherError` に変更は無い (0.3.0 と 0.4.0 の公開版で `stream.rs` / `hash.rs` / `kdf.rs` と依存は同一、0.4.1 は crate-type から `cdylib` を外しただけなので、暗号化した WAL と record の byte 列は変わらない)
- CI と `scripts/preflight.sh` の `NATIVE_FEATURES` に `analytics` と `crypto` を加えた それまで CI は `alice-crypto` を空の stub crate に置き換えており、`analytics` も対象外だったので、どちらの bridge も CI で compile されていなかった stub を作る `.github/actions/alice-stubs` は削除、`cargo hack` の除外は `python` だけにした `analytics_bridge` の clippy pedantic の指摘 (`# Errors` 節 5 件、test 3 件) を直した
- `alice-det-math` 0.4 is a direct dependency (the version `alice-zip` 0.9
  and `alice-analytics` 0.4 use). Bloom filter sizing takes its logarithm from `alice_det_math::ln64`
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

- **ALICE-CodeTracker v0.5.0** — the durability floor cleared here is the prerequisite for the `alice-tracker-alicedb` backend planned in ALICE-CodeTracker's CHANGELOG (0.4.0). Once alpha-2 stabilises we can wire the tracker against it.

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
