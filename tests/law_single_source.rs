//! 法則は 1 つ — `generate_all` / `query_point` / `query_range` は同じ bit を返す
//!
//! # なぜこの oracle が要るか
//!
//! 解析 model の復元則はこの crate に 2 つあった
//!
//! | 経路 | 使う法則 | 残差の適用 |
//! |---|---|---|
//! | `query_point` / `query_range` (解析 model) | `segment::law` = `f64` + platform libm | する |
//! | `generate_all` (`pub`) | `alice_core::generators` = `f32` | しない |
//!
//! 同一 segment を両方から読むと値が違う 同一入力・整数 index・`n = 1024` で上流の
//! 配列法則と `f64` の単点法則を突合すると、sine で **788 / 1024**、multi-sine で
//! **881 / 1024**、Fourier で **962 / 1024** の sample が不一致だった (2026-10-08 実測)
//! 原因は内部精度 (`f32` / `f64`)、DC の積む順、超越関数の出所の 3 つが重なったこと
//!
//! ⚠️ `generate_all` は `pub` なので、この食い違いは crate の外から観測できる
//!
//! # 残差はどちらに紐付いているか
//!
//! `MemTable::seal` は `let reconstructed = segment.query_point(timestamp)` に対して
//! `original.to_bits() ^ reconstructed.to_bits()` を保存する ⇒ **正典は単点経路**で、
//! `generate_all` が合っていなかった側である
//!
//! # この file が pin するもの
//!
//! 解析 model のすべてで、3 経路が **bit 一致**すること 許容差では比べない (旧実装の
//! 不一致はどれも 1e-6 以下なので、ゆるい許容差では 1 件も捕まらない)

use alice_db::model::ModelType;
use alice_db::segment::DataSegment;

/// `sample_pos` は `(ts - start) / (end - start) * (n - 1)` なので、`n - 1` が 2 の冪なら
/// 整数 timestamp が整数位置に厳密に写る (2 の冪での除算は丸めを起こさない)
///
/// ⚠️ これを守らないと「位置が整数でない」ことが原因の差を「法則が違う」と読んでしまう
const LENGTHS: [usize; 4] = [2, 5, 65, 129];

fn models(n: usize) -> Vec<(&'static str, ModelType)> {
    vec![
        (
            "SineWave",
            ModelType::SineWave {
                frequency: 3.0,
                amplitude: 2.0,
                phase: 0.4,
                offset: 0.25,
            },
        ),
        (
            "MultiSine",
            ModelType::MultiSine {
                components: vec![(1.0, 1.0, 0.0), (5.0, -0.5, 1.1), (13.0, 0.25, -2.2)],
                dc_offset: 0.125,
            },
        ),
        (
            "Fourier",
            ModelType::Fourier {
                // DC / Nyquist / 範囲外 k を含める (weight 分岐と skip を通す)
                coefficients: vec![(0, 10.0, 0.0), (1, 128.0, 0.3), (n / 2, 64.0, -0.9)],
                dc_offset: 0.3,
                sample_count: n,
            },
        ),
        (
            "Polynomial",
            ModelType::Polynomial {
                coefficients: vec![0.25, 1.5, -0.125, 0.0625],
                degree: 3,
                fit_error: 0.0,
            },
        ),
        (
            "Linear",
            ModelType::Linear {
                start_value: -3.0,
                end_value: 7.5,
            },
        ),
        ("Constant", ModelType::Constant { value: 1.75 }),
    ]
}

#[test]
fn the_three_read_paths_return_the_same_bits() {
    let mut compared = 0_usize;
    let mut mismatches: Vec<String> = Vec::new();

    for n in LENGTHS {
        let end = (n - 1) as i64;
        for (name, model) in models(n) {
            let segment = DataSegment::new(1, 0, end, model, n, n * 4);

            let all = segment.generate_all();
            assert_eq!(all.len(), n, "{name} n={n}: generate_all の長さが n でない");

            let range = segment.query_range(0, end);
            assert_eq!(
                range.len(),
                n,
                "{name} n={n}: query_range が {} 件 (n を期待)",
                range.len()
            );

            for i in 0..n {
                let ts = i as i64;
                let point = segment
                    .query_point(ts)
                    .unwrap_or_else(|| panic!("{name} n={n} i={i}: query_point が None"));
                let (rts, rval) = range[i];
                assert_eq!(
                    rts, ts,
                    "{name} n={n} i={i}: query_range の timestamp がずれた"
                );

                if all[i].to_bits() != point.to_bits() {
                    mismatches.push(format!(
                        "{name} n={n} i={i}: generate_all {} vs query_point {}",
                        all[i], point
                    ));
                }
                if rval.to_bits() != point.to_bits() {
                    mismatches.push(format!(
                        "{name} n={n} i={i}: query_range {rval} vs query_point {point}"
                    ));
                }
                compared += 1;
            }
        }
    }

    // ⚠️ 比較 0 件で通る経路を塞ぐ
    assert!(
        compared >= 1_000,
        "比較 {compared} 件 — 想定 (1000 件超) を下回った oracle が空振りしている"
    );

    // 内訳を出す ⚠️ 先頭 N 件だけ出すと「どの経路がどれだけ外れているか」が読めない
    let by_path = |needle: &str| mismatches.iter().filter(|m| m.contains(needle)).count();
    let by_model = |needle: &str| mismatches.iter().filter(|m| m.starts_with(needle)).count();
    assert!(
        mismatches.is_empty(),
        "3 経路が同じ bit を返していない ({} 件 / 比較 {compared} 件)\n\
         経路別: generate_all vs query_point {} 件 / query_range vs query_point {} 件\n\
         model 別: SineWave {} / MultiSine {} / Fourier {} / Polynomial {} / Linear {} / Constant {}\n{}",
        mismatches.len(),
        by_path("generate_all"),
        by_path("query_range"),
        by_model("SineWave"),
        by_model("MultiSine"),
        by_model("Fourier"),
        by_model("Polynomial"),
        by_model("Linear"),
        by_model("Constant"),
        mismatches
            .iter()
            .take(8)
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join("\n")
    );
}
