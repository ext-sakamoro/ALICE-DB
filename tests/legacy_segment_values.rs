//! Every segment file stored by an earlier release decodes to the values it held when it was written.
//!
//! oracle: `tests/fixtures/<store>/seg_1.values` was recorded from each fixture by the rkyv 0.7 reader that
//! wrote and read these files, before the stored segment format moved (see the CHANGELOG); it is the canonical
//! text of `tests/support/segment_text.rs` (every field, floats as bits). A reader that decodes any field
//! differently, or cannot decode a fixture, fails here.

#[path = "support/segment_text.rs"]
mod segment_text;

use alice_db::segment::DataSegment;
use std::path::Path;

const STORES: [&str; 5] = [
    "store_pre_0_3_0",
    "store_det_math_0_4",
    "store_db_0_5_0",
    "store_db_0_5_1",
    "store_db_0_6_0",
];

#[test]
fn every_stored_legacy_segment_decodes_to_its_recorded_values() {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut compared = 0;
    for store in STORES {
        let dir = fixtures.join(store);
        let bytes = std::fs::read(dir.join("seg_1.rkyv")).unwrap();
        let expected = std::fs::read_to_string(dir.join("seg_1.values")).unwrap();
        let seg = DataSegment::from_stored_bytes(&bytes).unwrap_or_else(|e| panic!("{store}: {e}"));
        assert_eq!(segment_text::segment_text(&seg), expected, "{store}");
        compared += 1;
    }
    assert_eq!(compared, STORES.len(), "compared nothing");
}
