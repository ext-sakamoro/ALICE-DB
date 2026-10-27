A lossless store written by `alice-db` 0.6.0 with `alice-det-math` 0.5.2 at 39507d1b and
`alice-zip` 0.11 at 8065587c (arithmetic identifier `50a40cf6…`), by the build of the commit that
adds this directory: the same writes as `store_db_0_6_0` (64 keys `0..=63`, value
`0.5 k + 0.125 ((7 k) mod 5)` (f32), one segment `seg_1.rkyv` whose residual is pinned to that
identifier, and one law record `ohm`, the line `1 + 2 x` fitted at `x = 0, 0.5, …, 4` with the
provenance `fixture` / `line`, indexed by its law identifier under it).

Against `store_db_0_6_0`, the decoded segment (`tests/support/segment_text.rs`) differs only in the
32-byte stamp inside the residual, and the decompressed law record only in its 32-byte identifier
(and the record checksum over it). The segment file is in the current stored format, which
`store_db_0_6_0` predates, so the files themselves are not compared byte for byte.

Read by `tests/identifier_history.rs` as a previous row of the identifier history and by
`tests/previous_identifier_store.rs`: alice-zip lists `50a40cf6…` as a previous identifier (the
move to `alice-det-math` 0.5.2 at 2d01c63f changed none of the functions it calls), so every value
reads back exact as `ReadStatus::VerifiedPrevious`.
