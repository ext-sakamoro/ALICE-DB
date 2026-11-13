//! A segment's residual is decoded up to the length its points need
//!
//! oracle (from the format, not read from the code): a residual holds one 4-byte correction per point, so
//! the decoder is given `4 * points` as its limit; a stream of exactly that length decodes back, one byte more
//! is refused (never read as the legacy raw form), and a read of such a segment reports it as corrupt. The
//! memory bound for a small stream that expands far past the limit is measured in
//! `tests/bounded_residual_memory.rs`.

use alice_db::segment::{
    compress_residual_xor, decompress_residual, residual_status, ResidualStatus,
};

fn words(n: u32) -> Vec<u8> {
    // compressible: a short repeating pattern, so the pinned LZMA form is taken
    (0..n).flat_map(|i| (i % 7).to_le_bytes()).collect()
}

/// `magic · LZMA alone stream of payload`, the form of the unpinned LZMA residuals
fn lzma_blob(magic: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = vec![magic];
    lzma_rs::lzma_compress(&mut std::io::Cursor::new(payload), &mut out).unwrap();
    out
}

#[test]
fn a_residual_of_exactly_the_needed_length_decodes_and_one_byte_more_is_refused() {
    let raw = words(4096);
    let blob = compress_residual_xor(&raw, &alice_db::SEMANTICS_ID);
    assert!(blob.len() < raw.len(), "the LZMA path was not taken");
    assert_eq!(decompress_residual(&blob, raw.len()), Some(raw.clone()));
    assert_eq!(decompress_residual(&blob, raw.len() - 1), None);
}

#[test]
fn every_lzma_form_is_bounded() {
    let raw = words(1024);
    // 0: the earliest LZMA form, 2: unpinned xor LZMA
    for magic in [0u8, 2] {
        let blob = lzma_blob(magic, &raw);
        assert_eq!(
            decompress_residual(&blob, raw.len()),
            Some(raw.clone()),
            "magic {magic}"
        );
        assert_eq!(
            decompress_residual(&blob, raw.len() - 1),
            None,
            "magic {magic}: a stream past the limit must not fall back to the raw form"
        );
    }
}

// oracle: when the stream's dictionary (8 MiB from this writer) is smaller than the limit, the decoder's window
// wraps and hands its bytes to the output, and the output's limit stops it there
#[test]
fn a_stream_with_a_dictionary_below_the_limit_is_stopped_at_the_output() {
    let raw = vec![0u8; 9 << 20];
    let blob = lzma_blob(2, &raw);
    assert_eq!(
        u32::from_le_bytes(blob[2..6].try_into().unwrap()),
        8 << 20,
        "the writer's dictionary"
    );
    assert_eq!(
        decompress_residual(&blob, raw.len()).map(|v| v.len()),
        Some(raw.len())
    );
    assert_eq!(decompress_residual(&blob, (8 << 20) + (512 << 10)), None);
}

#[test]
fn a_segment_whose_residual_expands_past_its_points_is_corrupt() {
    let raw = words(4096);
    let blob = compress_residual_xor(&raw, &alice_db::SEMANTICS_ID);
    assert_eq!(residual_status(Some(&blob), 4096), ResidualStatus::Exact);
    match residual_status(Some(&blob), 4095) {
        ResidualStatus::Corrupt {
            needed_bytes,
            found_bytes,
        } => {
            assert_eq!(needed_bytes, 4095 * 4);
            assert!(found_bytes > needed_bytes, "{found_bytes}");
        }
        other => panic!("{other:?}"),
    }
}
