//! Fuzz target: arbitrary bytes as a stored law record
//!
//! `decode_law_record` must return `Ok` or `Err` without panicking or
//! allocating for counts the input cannot hold. A record that decodes is
//! canonical: encoding it again gives the same bytes. Restoring it with
//! `SignalLaw::from_parts` must not panic either.

#![no_main]

use alice_db::law_store::{decode_law_record, encode_law_record, SignalLaw};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok((name, version, parts)) = decode_law_record(data) {
        let again = encode_law_record(&name, version, &parts).expect("decoded record re-encodes");
        assert_eq!(again, data, "a decoded record is canonical");
        let _ = SignalLaw::from_parts(parts);
    }
});
