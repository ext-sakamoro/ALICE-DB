//! Fuzz target: arbitrary bytes as a stored segment file
//!
//! `DataSegment::from_stored_bytes` must return `Ok` or `Err` without panicking or allocating for lengths the
//! input cannot hold, for both stored formats: each input is decoded as it is (no header: the format 0
//! reader of the rkyv 0.7 layout) and behind a format 1 header (the rkyv 0.8 archive). A segment that
//! decodes is written again in the current format and must decode to the same segment.

#![no_main]

use alice_db::segment::{DataSegment, SEGMENT_FORMAT};
use libfuzzer_sys::fuzz_target;

fn check(bytes: &[u8]) {
    if let Ok(seg) = DataSegment::from_stored_bytes(bytes) {
        let again = seg.to_rkyv_bytes().expect("a decoded segment encodes");
        let back = DataSegment::from_stored_bytes(&again).expect("what was written decodes");
        assert_eq!(format!("{seg:?}"), format!("{back:?}"));
    }
}

fuzz_target!(|data: &[u8]| {
    check(data);
    let mut framed = Vec::with_capacity(16 + data.len());
    framed.extend_from_slice(b"ALICESEG");
    framed.extend_from_slice(&SEGMENT_FORMAT.to_le_bytes());
    framed.extend_from_slice(&[0; 4]);
    framed.extend_from_slice(data);
    check(&framed);
});
