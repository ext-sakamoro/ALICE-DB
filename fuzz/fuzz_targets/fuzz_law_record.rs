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
//! well-formed envelope around arbitrary body bytes, once with the current
//! format (which carries a semantics identifier) and once with format 1
//! (which does not), which is what fuzzes the parser behind the checksum.
//!
//! The encoder writes only the current format, so a format 1 record is
//! checked against it: the current encoding with the 32 identifier bytes
//! removed, format 1 and a new CRC is the same record.

#![no_main]

use alice_db::law_store::{
    decode_law_record, encode_law_record, SignalLaw, LAW_RECORD_FORMAT, LAW_RECORD_MAGIC,
};
use libfuzzer_sys::fuzz_target;

/// magic 8 + format 4
const HEADER: usize = 12;

fn check(record: &[u8]) {
    let Ok((name, version, parts, semantics)) = decode_law_record(record) else {
        return;
    };
    match semantics {
        Some(id) => {
            let again =
                encode_law_record(&name, version, &parts, &id).expect("decoded record re-encodes");
            assert_eq!(again, record, "a decoded record is canonical");
        }
        None => {
            let v2 = encode_law_record(&name, version, &parts, &[0; 32])
                .expect("decoded record re-encodes");
            // the identifier follows name length 4, name and version 8
            let at = HEADER + 4 + name.len() + 8;
            let mut v1 = v2[..v2.len() - 4].to_vec();
            v1.drain(at..at + 32);
            v1[8..12].copy_from_slice(&1u32.to_le_bytes());
            let crc = crc32fast::hash(&v1);
            v1.extend_from_slice(&crc.to_le_bytes());
            assert_eq!(v1, record, "a decoded format 1 record is canonical");
        }
    }
    let _ = SignalLaw::from_parts(parts);
}

fn wrapped(format: u32, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(body.len() + 16);
    out.extend_from_slice(&LAW_RECORD_MAGIC);
    out.extend_from_slice(&format.to_le_bytes());
    out.extend_from_slice(body);
    let crc = crc32fast::hash(&out);
    out.extend_from_slice(&crc.to_le_bytes());
    out
}

fuzz_target!(|data: &[u8]| {
    check(data);
    check(&wrapped(LAW_RECORD_FORMAT, data));
    check(&wrapped(1, data));
});
