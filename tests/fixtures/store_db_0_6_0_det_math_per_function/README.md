A lossless store written by `alice-db` 0.6.0 with `alice-zip` 0.11 at its identifier `65f48962…`
(`alice-det-math` folded per function, the functions `alice-zip` calls), by the build of the commit
that adds this directory: the same writes as `store_db_0_6_0` (64 keys `0..=63`, value
`0.5 k + 0.125 ((7 k) mod 5)` (f32), one segment `seg_1.rkyv` whose residual is pinned to that
identifier, and one law record `ohm`, the line `1 + 2 x` fitted at `x = 0, 0.5, …, 4` with the
provenance `fixture` / `line`, indexed by its law identifier under it).

Against `store_db_0_6_0_det_math_2d01c63f`, `seg_1.rkyv` differs in exactly the 32 bytes of the stamp
inside the residual, and `index.alice` and `data.alice` are the same bytes; the law record restores
the same law with only its identifier and the record checksum over it changed; the record is
compressed, so its stored length differs by a byte (305 bytes against 304). The store's lock file
`blob.lock` (written by the open since the one-store rule) is not committed: it is not part of
the data and is written again by every open.

Read by `tests/identifier_history.rs` as the current row of the identifier history: every value
reads back exact as `ReadStatus::Current`.
