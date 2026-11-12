A lossless store written by `alice-db` 0.6.0 with `alice-zip` 0.11 at its identifier `922b2865…`
(`alice-det-math` folded per function and the kind tag of `alice-zip`'s causal 2-D law folded as an
entry), by the build of the commit that adds this directory: the same writes as `store_db_0_6_0`
(64 keys `0..=63`, value `0.5 k + 0.125 ((7 k) mod 5)` (f32), one segment `seg_1.rkyv` whose
residual is pinned to that identifier, and one law record `ohm`, the same law as in
`store_db_0_6_0_det_math_per_function`, indexed by its law identifier under `922b2865…`).

Against `store_db_0_6_0_det_math_per_function`, `seg_1.rkyv` differs in exactly the 32 bytes of the
stamp inside the residual (bytes 33 to 64), and `index.alice` and `data.alice` are the same bytes;
the law record restores the same law with only its identifier and the record checksum over it
changed; the record is compressed, so its stored length differs (303 bytes against 305). The
store's lock file `blob.lock` is not committed: it is not part of the data and is written again
by every open.

Read by `tests/identifier_history.rs` as the current row of the identifier history: every value
reads back exact as `ReadStatus::Current`.
