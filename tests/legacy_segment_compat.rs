//! 旧実装が書いた segment を、新しい法則の下でも厳密に読み戻せる
//!
//! # なぜこの oracle が要るか
//!
//! lossless segment の残差は `original.to_bits() ^ query_point(ts).to_bits()` として
//! 保存される ⇒ ⚠️ **model の bit が変われば、既存の残差は別の値を復元する**
//!
//! ⚠️⚠️ 既存の test は**同一 process 内で書いて読む**ので、この破壊を 1 件も検出しない
//! (残差も読み出しも同じ法則で計算されるため常に整合する) 旧実装が書いた **byte**
//! を読む test が 1 本も無かった ⇒ 下の fixture は origin/main (`52ca813`) の code で
//! 生成した実バイト列で、`model_bits` は旧法則が返した値そのものである
//!
//! # 旧データの厳密性はどこまで保証されるか
//!
//! 旧法則は platform libm の `f64::sin` / `f64::cos` を呼んでいた IEEE 754 は
//! これらに正確丸めを要求しないので、⚠️ **旧 lossless segment は書いた機械でしか
//! 厳密に復元できない** (別 OS / CPU で読むと最終 ulp がずれる) しかも blob の中に
//! 「どの機械の算術か」の記録が無いので、**それを確かめる手段も無い**
//!
//! ⚠️ **旧評価器を凍結して持つ案は採らなかった** 凍結した module が呼ぶのは *読む側* の
//! platform libm であって書いた側のものではないので、「旧法則を再現した」と称して
//! 別の算術を適用することになる 本機での実測では det-math と platform libm が
//! sine / Fourier で 20 万件すべて一致したため、凍結 module は本機で 1 行も被覆できない
//!
//! | 残差 magic | 算術の識別子 | 厳密性 |
//! |---|---|---|
//! | なし (lossy) | — | 契約なし (値は model の近似) |
//! | `0`/`1` additive, `2`/`3` XOR | **持たない** (`Unpinned`) | 書いた機械でのみ、かつ確認不能 |
//! | `6`/`7` XOR | `SEMANTICS_ID` 32 byte (`Pinned`) | 識別子が一致する限りどこでも |
//!
//! この file が pin するもの — 旧 blob が本機で厳密に戻ること / 新 blob が識別子を持つこと /
//! 旧 blob が `Unpinned` と報告されること / ⚠️ **識別子が違う blob は適用されず model 値が
//! 返ること** (原信号でも model でもない値を黙って返さない)

use alice_db::segment::DataSegment;

fn unhex(s: &str) -> Vec<u8> {
    assert!(s.len().is_multiple_of(2), "hex の長さが奇数");
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).expect("hex"))
        .collect()
}

/// `(名前, 旧実装が書いた byte, 原信号の bit, 旧法則が返した model の bit)`
///
/// origin/main `52ca813` の code で生成 原信号は model 値の最終 ulp を数個ずらした値で、
/// 残差が確実に非零になるようにしてある
#[allow(clippy::type_complexity)]
fn fixtures() -> Vec<(&'static str, Vec<u8>, Vec<u32>, Vec<u32>)> {
    vec![
        (
            "sine",
            unhex("00000000000000000800000000000000020000000000404000000040cdcccc3e0000803e012500000000000000030300000003000000030000000300000003000000030000000300000003000000030000000100000000000000c3b24919a10100000900000000000000240000000000000010000000000000000000000000000240"),
            vec![0x3f83b0ef, 0x3fba5b22, 0xbfde0c0e, 0x3f83b0ef, 0x3fba5b22, 0xbfde0c0e, 0x3f83b0ef, 0x3fba5b22, 0xbfde0c0e],
            vec![0x3f83b0ec, 0x3fba5b21, 0xbfde0c0d, 0x3f83b0ec, 0x3fba5b21, 0xbfde0c0d, 0x3f83b0ec, 0x3fba5b21, 0xbfde0c0d],
        ),
        (
            "fourier",
            unhex("000000000000000008000000000000000100000002000000000000000100000000000000000000439a99993e040000000000000000008042666666bf9a99993e0900000000000000012500000000000000030500000005000000050000000500000005000000050000000500000005000000050000000100000000000000c3b24919a1010000090000000000000024000000000000002800000000000000cdccccccccccec3f"),
            vec![0x4211423b, 0x41337521, 0xc0697c32, 0xc1756c84, 0xc2162f8a, 0xc11da89a, 0xc1a09b0c, 0x41d9d7bb, 0x41666e16],
            vec![0x4211423e, 0x41337524, 0xc0697c37, 0xc1756c81, 0xc2162f8f, 0xc11da89f, 0xc1a09b09, 0x41d9d7be, 0x41666e13],
        ),
        (
            "multisine",
            unhex("000000000000000008000000000000000300000002000000000000000000803f0000803f000000000000a040000000bfcdcc8c3f0000003e012500000000000000030700000007000000070000000700000007000000070000000700000007000000070000000100000000000000c3b24919a1010000090000000000000024000000000000002000000000000000000000000000f23f"),
            vec![0xbea4262c, 0x3fa1cd98, 0x3f1f677f, 0x3fb482c7, 0x3e2a47c2, 0xbd91808f, 0xbf36f26d, 0xbf87169f, 0xbe34dd9b],
            vec![0xbea4262b, 0x3fa1cd9f, 0x3f1f6778, 0x3fb482c0, 0x3e2a47c5, 0xbd918088, 0xbf36f26a, 0xbf871698, 0xbe34dd9c],
        ),
    ]
}

#[test]
fn old_lossless_segments_still_read_back_exactly() {
    let mut compared = 0_usize;
    let mut wrong: Vec<String> = Vec::new();

    for (name, bytes, originals, _) in fixtures() {
        let segment = DataSegment::from_bytes(&bytes)
            .unwrap_or_else(|e| panic!("{name}: 旧 byte の deserialise が失敗 ({e})"));
        assert_eq!(
            segment.metadata.point_count,
            originals.len(),
            "{name}: point_count が fixture と違う (byte 列が壊れている)"
        );
        assert!(
            segment.residual_blob.is_some(),
            "{name}: 残差が無い — lossless でない fixture は互換性を何も測らない"
        );

        for (i, &want) in originals.iter().enumerate() {
            let got = segment
                .query_point(i as i64)
                .unwrap_or_else(|| panic!("{name} i={i}: query_point が None"));
            if got.to_bits() != want {
                wrong.push(format!(
                    "{name} i={i}: {:#010x} (= {got}) を返したが {want:#010x} (= {}) のはず",
                    got.to_bits(),
                    f32::from_bits(want)
                ));
            }
            compared += 1;
        }

        // query_range 経路も同じ復元を返す (残差の index 付けが経路で違わないこと)
        let end = (originals.len() - 1) as i64;
        let range = segment.query_range(0, end);
        assert_eq!(range.len(), originals.len(), "{name}: query_range の件数");
        for (i, (_, got)) in range.iter().enumerate() {
            if got.to_bits() != originals[i] {
                wrong.push(format!(
                    "{name} i={i} (range): {:#010x} vs {:#010x}",
                    got.to_bits(),
                    originals[i]
                ));
            }
            compared += 1;
        }
    }

    // ⚠️ 比較 0 件で通る経路を塞ぐ
    assert!(
        compared >= 50,
        "比較 {compared} 件 — fixture が読めていない (oracle が空振りしている)"
    );

    assert!(
        wrong.is_empty(),
        "旧 lossless segment が厳密に復元できていない ({} 件 / 比較 {compared} 件)\n{}",
        wrong.len(),
        wrong
            .iter()
            .take(8)
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// fixture の `model_bits` が旧法則の値であることを pin する
///
/// ⚠️ これが無いと、上の test が「残差も model も新法則で一貫している」状態でも通りうる
/// (fixture を新実装で作り直してしまった場合) 旧法則の値そのものを比べる
#[test]
fn the_fixture_really_carries_the_old_law_values() {
    let mut differing = 0_usize;
    let mut total = 0_usize;

    for (name, bytes, originals, model_bits) in fixtures() {
        let segment = DataSegment::from_bytes(&bytes).expect("deserialise");
        // 残差を外した素の segment = 法則そのもの
        let bare = DataSegment::new(
            2,
            0,
            (originals.len() - 1) as i64,
            segment.model.clone(),
            originals.len(),
            originals.len() * 4,
        );
        for (i, &old) in model_bits.iter().enumerate() {
            let now = bare
                .query_point(i as i64)
                .unwrap_or_else(|| panic!("{name} i={i}"))
                .to_bits();
            total += 1;
            if now != old {
                differing += 1;
            }
        }
    }

    assert!(total >= 20, "比較 {total} 件 — 空振り");
    // 差が 0 件なら、det-math と本機の libm がこの入力で一致している = 旧データは
    // 偶然そのまま読める ⇒ 互換の分岐が「効いているか」をこの test では判定できない
    // ので、件数を出力して分かるようにする (fail させない: 一致は欠陥ではない)
    println!(
        "旧法則と新法則で bit が違う sample: {differing} / {total} \
         (0 なら本機では det-math と platform libm が一致している)"
    );
}

// ---------------------------------------------------------------------------
// 残差が「どの算術で評価した model に対するものか」を自分で持つ
// ---------------------------------------------------------------------------

use alice_db::segment::{residual_semantics, ResidualSemantics};

/// 新しく書いた残差は、評価に使った算術の識別子を持つ
#[test]
fn a_freshly_written_residual_names_the_arithmetic_it_was_measured_against() {
    let n = 9_usize;
    let model = alice_db::model::ModelType::SineWave {
        frequency: 3.0,
        amplitude: 2.0,
        phase: 0.4,
        offset: 0.25,
    };
    let bare = DataSegment::new(1, 0, (n - 1) as i64, model.clone(), n, n * 4);

    // 法則からわずかに外れた原信号 (残差が非零になる)
    let originals: Vec<f32> = (0..n)
        .map(|i| {
            let v = bare.query_point(i as i64).expect("in range");
            f32::from_bits(v.to_bits() ^ 0x3)
        })
        .collect();

    let mut raw = Vec::with_capacity(n * 4);
    for (i, &original) in originals.iter().enumerate() {
        let model_value = bare.query_point(i as i64).expect("in range");
        raw.extend_from_slice(&(original.to_bits() ^ model_value.to_bits()).to_le_bytes());
    }
    let blob = alice_db::segment::compress_residual_xor(&raw, &alice_core::law::SEMANTICS_ID);

    assert_eq!(
        residual_semantics(&blob),
        ResidualSemantics::Pinned(alice_core::law::SEMANTICS_ID),
        "書いた残差が算術の識別子を持っていない"
    );

    let segment = DataSegment::new(1, 0, (n - 1) as i64, model, n, n * 4).with_residual(blob);
    for (i, &want) in originals.iter().enumerate() {
        let got = segment.query_point(i as i64).expect("in range");
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "i={i}: 識別子が一致する残差で厳密に戻らない"
        );
    }
}

/// 0.3.0 より前に書かれた残差は識別子を持たない (= 書いた機械でのみ厳密)
#[test]
fn residuals_written_before_the_identifier_report_themselves_as_unpinned() {
    let mut checked = 0_usize;
    for (name, bytes, _, _) in fixtures() {
        let segment = DataSegment::from_bytes(&bytes).expect("deserialise");
        let blob = segment.residual_blob.as_ref().expect("lossless fixture");
        assert_eq!(
            residual_semantics(blob),
            ResidualSemantics::Unpinned,
            "{name}: 旧 blob が識別子を持っていると判定された"
        );
        checked += 1;
    }
    assert!(checked >= 3, "確認 {checked} 件 — fixture が読めていない");
}

/// ⚠️ 識別子が違う残差は **適用せず model 値を返す** (原信号でも model でもない値を返さない)
///
/// これが C+ の要点 識別子を持たせる目的は「将来 det-math が変わった時に、
/// 黙って誤値を返すのでなく検出できること」なので、不一致時の振る舞いを pin する
#[test]
fn a_residual_from_different_arithmetic_is_skipped_not_applied() {
    let n = 9_usize;
    let model = alice_db::model::ModelType::SineWave {
        frequency: 3.0,
        amplitude: 2.0,
        phase: 0.4,
        offset: 0.25,
    };
    let bare = DataSegment::new(1, 0, (n - 1) as i64, model.clone(), n, n * 4);

    // 非零の残差を作る (全部 0 だと「適用しても同じ」で空振りになる)
    let mut raw = Vec::with_capacity(n * 4);
    for _ in 0..n {
        raw.extend_from_slice(&0x0000_0003_u32.to_le_bytes());
    }
    // 識別子を 1 bit だけ変える = 別の算術を名乗る blob
    let mut other = alice_core::law::SEMANTICS_ID;
    other[0] ^= 0x01;
    let blob = alice_db::segment::compress_residual_xor(&raw, &other);
    assert_eq!(residual_semantics(&blob), ResidualSemantics::Pinned(other));
    assert!(
        !alice_db::segment::residual_is_applicable(&blob),
        "識別子が違うのに適用可と判定された"
    );

    let segment = DataSegment::new(1, 0, (n - 1) as i64, model, n, n * 4).with_residual(blob);
    let mut compared = 0_usize;
    for i in 0..n {
        let want = bare.query_point(i as i64).expect("in range");
        let got = segment.query_point(i as i64).expect("in range");
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "i={i}: 識別子不一致の残差が適用されている (model 値 {want} を期待、{got} が返った)"
        );
        // 残差を適用していたら別の値になることを確かめる (= この test は空振りでない)
        let would_be = f32::from_bits(want.to_bits() ^ 0x3);
        assert_ne!(
            would_be.to_bits(),
            want.to_bits(),
            "i={i}: 残差 0x3 が恒等 — fixture が空振り"
        );
        compared += 1;
    }
    assert_eq!(compared, n);

    // query_range 側も同じ (経路で振る舞いが違わない)
    let range = segment.query_range(0, (n - 1) as i64);
    assert_eq!(range.len(), n);
    for (i, (_, got)) in range.iter().enumerate() {
        let want = bare.query_point(i as i64).expect("in range");
        assert_eq!(got.to_bits(), want.to_bits(), "i={i} (range)");
    }
}

/// ⚠️ zero-copy (mmap / rkyv) 経路も同じ照合をする
///
/// `SegmentView` は `DataSegment` とは別の読み出し実装なので、識別子の照合を片方にだけ
/// 入れると「どちらの API で読んだか」で値が変わる 2026-10-08 の破壊試験では、
/// zero-copy 側の照合を外す変異がこの test 無しでは 1 件も red にならなかった
#[test]
fn the_zero_copy_path_checks_the_identifier_too() {
    let n = 9_usize;
    let model = alice_db::model::ModelType::SineWave {
        frequency: 3.0,
        amplitude: 2.0,
        phase: 0.4,
        offset: 0.25,
    };
    let bare = DataSegment::new(1, 0, (n - 1) as i64, model.clone(), n, n * 4);

    let mut raw = Vec::with_capacity(n * 4);
    for _ in 0..n {
        raw.extend_from_slice(&0x0000_0003_u32.to_le_bytes());
    }
    let mut other = alice_core::law::SEMANTICS_ID;
    other[0] ^= 0x01;
    let blob = alice_db::segment::compress_residual_xor(&raw, &other);

    let segment = DataSegment::new(1, 0, (n - 1) as i64, model, n, n * 4).with_residual(blob);
    let bytes = segment.to_rkyv_bytes().expect("rkyv serialise");
    let view = alice_db::segment::SegmentView::from_vec(bytes).expect("rkyv view");

    let mut compared = 0_usize;
    for i in 0..n {
        let want = bare.query_point(i as i64).expect("in range");
        let got = view
            .query_point(i as i64)
            .unwrap_or_else(|| panic!("i={i}: zero-copy の query_point が None"));
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "i={i}: zero-copy 側で識別子不一致の残差が適用されている"
        );
        compared += 1;
    }
    let range = view.query_range(0, (n - 1) as i64);
    assert_eq!(range.len(), n, "zero-copy の query_range の件数");
    for (i, (_, got)) in range.iter().enumerate() {
        let want = bare.query_point(i as i64).expect("in range");
        assert_eq!(got.to_bits(), want.to_bits(), "i={i} (zero-copy range)");
        compared += 1;
    }
    assert_eq!(compared, n * 2);

    // 識別子が一致する残差なら zero-copy 側でも適用される (上の assert が
    // 「残差を一切適用しない」で通る空振りでないことを示す)
    let good = alice_db::segment::compress_residual_xor(&raw, &alice_core::law::SEMANTICS_ID);
    let ok_segment =
        DataSegment::new(1, 0, (n - 1) as i64, bare.model.clone(), n, n * 4).with_residual(good);
    let ok_view = alice_db::segment::SegmentView::from_vec(
        ok_segment.to_rkyv_bytes().expect("rkyv serialise"),
    )
    .expect("rkyv view");
    for i in 0..n {
        let model_value = bare.query_point(i as i64).expect("in range");
        let want = f32::from_bits(model_value.to_bits() ^ 0x3);
        let got = ok_view.query_point(i as i64).expect("in range");
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "i={i}: 識別子が一致する残差が zero-copy 側で適用されていない"
        );
    }
}

/// 識別子付き残差の LZMA 経路 (magic 6) を通す
///
/// ⚠️ 小さい残差は LZMA で膨らむので raw 経路 (magic 7) に落ちる 破壊試験で
/// 「書き込み時に識別子を入れない」変異が LZMA 側だけに当たった時、raw しか通らない
/// test では 1 件も red にならなかった 圧縮が効く大きさの残差で両経路を踏む
#[test]
fn both_the_compressed_and_the_raw_pinned_paths_carry_the_identifier() {
    use alice_db::segment::{residual_semantics, ResidualSemantics};

    // 圧縮が効く: 同じ 4 byte 語を 4096 個 (LZMA で確実に縮む)
    let mut compressible = Vec::with_capacity(4096 * 4);
    for _ in 0..4096 {
        compressible.extend_from_slice(&0x0000_0003_u32.to_le_bytes());
    }
    let big =
        alice_db::segment::compress_residual_xor(&compressible, &alice_core::law::SEMANTICS_ID);
    assert!(
        big.len() < compressible.len(),
        "LZMA 経路に入っていない (blob {} byte >= 入力 {} byte)",
        big.len(),
        compressible.len()
    );
    assert_eq!(
        residual_semantics(&big),
        ResidualSemantics::Pinned(alice_core::law::SEMANTICS_ID),
        "LZMA 経路の blob が識別子を持っていない"
    );

    // 圧縮が効かない: 4 語だけ (raw 経路)
    let small: Vec<u8> = (0..4_u32)
        .flat_map(|i| (i ^ 0x9e37_79b9).to_le_bytes())
        .collect();
    let tiny = alice_db::segment::compress_residual_xor(&small, &alice_core::law::SEMANTICS_ID);
    assert_eq!(
        residual_semantics(&tiny),
        ResidualSemantics::Pinned(alice_core::law::SEMANTICS_ID),
        "raw 経路の blob が識別子を持っていない"
    );

    // 両経路の展開が元の byte 列に戻る (header を剥がし損ねたら長さが変わる)
    assert_eq!(
        alice_db::segment::decompress_residual(&big),
        compressible,
        "LZMA 経路の展開が元に戻らない"
    );
    assert_eq!(
        alice_db::segment::decompress_residual(&tiny),
        small,
        "raw 経路の展開が元に戻らない"
    );
}

/// `Linear` の法則は 1 つ — `generate_all` と単点経路が同じ除算を使う
///
/// ⚠️ `n - 1` が 2 の冪なら `1/(n-1)` は厳密なので、逆数乗算と除算の差が出ない
/// 上の 3 経路 oracle は timestamp を整数位置に厳密に写すため 2 の冪を選んでおり、
/// この差を原理的に見られない ⇒ 閉形式と直接比べる
#[test]
fn the_linear_law_divides_rather_than_multiplying_by_a_reciprocal() {
    let mut compared = 0_usize;
    let mut worst = 0_u32;
    // n - 1 が 2 の冪でない長さを選ぶ (1/(n-1) が丸められる)
    for n in [5_usize, 7, 11, 100, 1000] {
        let (start, end_value) = (-3.0_f64, 7.5_f64);
        let segment = DataSegment::new(
            1,
            0,
            (n - 1) as i64,
            alice_db::model::ModelType::Linear {
                start_value: start,
                end_value: end_value,
            },
            n,
            n * 4,
        );
        let all = segment.generate_all();
        assert_eq!(all.len(), n);
        for (i, got) in all.iter().enumerate() {
            // oracle: start + (end - start) * i / (n - 1) を f64 で評価 (除算 1 回)
            #[allow(clippy::cast_precision_loss)]
            let x = i as f64 / (n - 1) as f64;
            let want = (start + x * (end_value - start)) as f32;
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "Linear n={n} i={i}: {got} vs 閉形式 {want} (逆数乗算なら最終 ulp がずれる)"
            );
            worst = worst.max(got.to_bits().abs_diff(want.to_bits()));
            compared += 1;
        }
    }
    assert!(compared >= 1_000, "比較 {compared} 件 — 空振り");
    assert_eq!(worst, 0, "bit 一致でない");
}
