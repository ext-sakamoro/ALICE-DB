A lossless store written by `alice-db` 0.6.0 (`2789c41`, before its release; `alice-zip`
0.11, arithmetic identifier `51ffeb07…`): 64 keys `0..=63`, value
`0.5 k + 0.125 ((7 k) mod 5)` (f32), one segment `seg_1.rkyv` whose residual is pinned
to that identifier, and one law record `ohm` (a line fitted to `2 x + 1`).

Read by `tests/identifier_history.rs` as a previous row of the identifier history and by
`tests/previous_identifier_store.rs`: alice-zip lists `51ffeb07…` as a previous identifier
(the move to `alice-det-math` 0.5.2 at 39507d1b changed none of the functions it calls), so
every value reads back exact as `ReadStatus::VerifiedPrevious`.
