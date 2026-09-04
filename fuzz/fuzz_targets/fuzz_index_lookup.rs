//! Fuzz target: 攻撃者制御 index key で任意 key lookup が panic せず終端することを検証
//!
//! ALICE-DB v0.1.0 では public index API が未 export のため、本 target は
//! canonical scaffold として byte slice 走査のみを行う。
//! index API 復活後は `alice_db::index::lookup(&idx, &key)` 等の呼び出しへ差し替える。
//!
//! canonical CI template [[reference_alice_ci_canonical_template]] 準拠

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // 巨大 input による fuzzer timeout 回避
    if data.len() > 64 * 1024 {
        return;
    }
    // 現段階では走査自体で panic せず終端することを保証
    let mut sum: u64 = 0;
    for &b in data {
        sum = sum.wrapping_add(b as u64);
    }
    // index lookup 実装後は以下のような呼び出しに差し替え:
    //   let idx = alice_db::index::Index::default();
    //   let _ = idx.lookup(data);
    let _ = sum;
});
