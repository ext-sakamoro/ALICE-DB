//! Fuzz target: 攻撃者制御 byte 列を query 文字列として解釈する経路が panic しないことを検証
//!
//! ALICE-DB v0.1.0 では public query parser が未 export のため、本 target は
//! canonical scaffold として bytes → UTF-8 lossy 変換のみを行う。
//! query parser 復活後は `alice_db::query::parse(&str)` 等の呼び出しへ差し替える。
//!
//! canonical CI template [[reference_alice_ci_canonical_template]] 準拠

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // 巨大 input による fuzzer timeout 回避
    if data.len() > 64 * 1024 {
        return;
    }
    // UTF-8 lossy 変換自体で panic せず終端することを保証
    let _s = String::from_utf8_lossy(data);
    // query parser 実装後は以下のような呼び出しに差し替え:
    //   let _ = alice_db::query::parse(&_s);
});
