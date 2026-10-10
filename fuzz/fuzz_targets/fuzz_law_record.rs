//! Fuzz target: arbitrary bytes as a stored law record
//!
//! `decode_law_record` must return `Ok` or `Err` without panicking or
//! allocating for counts the input cannot hold. A record that decodes is
//! canonical: encoding it again gives the same bytes. Restoring it with
//! `SignalLaw::from_parts` must not panic either.
//!
//! A record ends in a CRC32 of everything before it, so a mutated input almost
//! never reaches the body parser: it stops at "checksum mismatch". Each input
//! is therefore also decoded wrapped as `magic ‖ format ‖ input ‖ crc32`, a
//! well-formed envelope around arbitrary body bytes, which is what fuzzes the
//! parser behind the checksum.

#![no_main]

use alice_db::law_store::{
    decode_law_record, encode_law_record, SignalLaw, LAW_RECORD_FORMAT, LAW_RECORD_MAGIC,
};
use libfuzzer_sys::fuzz_target;

fn check(record: &[u8]) {
    if let Ok((name, version, parts)) = decode_law_record(record) {
        let again = encode_law_record(&name, version, &parts).expect("decoded record re-encodes");
        assert_eq!(again, record, "a decoded record is canonical");
        let _ = SignalLaw::from_parts(parts);
    }
}

fuzz_target!(|data: &[u8]| {
    check(data);
    let mut wrapped = Vec::with_capacity(data.len() + 16);
    wrapped.extend_from_slice(&LAW_RECORD_MAGIC);
    wrapped.extend_from_slice(&LAW_RECORD_FORMAT.to_le_bytes());
    wrapped.extend_from_slice(data);
    let crc = crc32fast::hash(&wrapped);
    wrapped.extend_from_slice(&crc.to_le_bytes());
    check(&wrapped);
});
