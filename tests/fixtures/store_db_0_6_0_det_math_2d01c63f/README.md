A lossless store written by `alice-db` 0.6.0 with `alice-det-math` 0.5.2 at 2d01c63f and
`alice-zip` 0.11 at 19f6f2c1 (arithmetic identifier `17d0df1d…`), by the build of the commit that
adds this directory: the same writes as `store_db_0_6_0` (64 keys `0..=63`, value
`0.5 k + 0.125 ((7 k) mod 5)` (f32), one segment `seg_1.rkyv` whose residual is pinned to that
identifier, and one law record `ohm`, the line `1 + 2 x` fitted at `x = 0, 0.5, …, 4` with the
provenance `fixture` / `line`, indexed by its law identifier under it).

Against `store_db_0_6_0_det_math_0_5_2`, `seg_1.rkyv` differs in exactly the 32 bytes of the stamp
inside the residual, and the law record restores the same law (its parts compare equal) with only
its identifier and the record checksum over it changed; the record is compressed, so its stored
length can differ by a byte.

Read by `tests/identifier_history.rs` as the current row of the identifier history: every value
reads back exact as `ReadStatus::Current`.
