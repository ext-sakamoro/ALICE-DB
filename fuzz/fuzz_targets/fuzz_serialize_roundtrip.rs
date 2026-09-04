//! Fuzz target: 攻撃者制御 byte 列で row serialize / deserialize round-trip が panic せず終端することを検証
//!
//! ALICE-DB v0.1.0 では public row codec が未 export のため、本 target は
//! canonical scaffold として arbitrary 由来の入力を受けて何もせず終端する。
//! row codec 復活後は `alice_db::model::Row::decode(&bytes)` → `encode()` の
//! round-trip 検査へ差し替える。
//!
//! canonical CI template [[reference_alice_ci_canonical_template]] 準拠

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct Input {
    key: Vec<u8>,
    value: Vec<u8>,
}

fuzz_target!(|input: Input| {
    // 巨大 input による fuzzer timeout 回避
    if input.key.len() > 8 * 1024 || input.value.len() > 64 * 1024 {
        return;
    }
    // round-trip API 実装後は以下のような呼び出しに差し替え:
    //   let row = alice_db::model::Row::new(&input.key, &input.value);
    //   let bytes = row.encode();
    //   let decoded = alice_db::model::Row::decode(&bytes).unwrap();
    //   assert_eq!(decoded, row);
    let _ = (&input.key, &input.value);
});
