A lossless store written by the published `alice-db` 0.5.0 (`alice-zip` 0.10.0,
`alice-det-math` 0.5.0, arithmetic identifier `135c407d…`): 64 keys `0..=63`,
value `0.5 k + 0.125 ((7 k) mod 5)` (f32), one segment `seg_1.rkyv` whose
residual is pinned to that identifier, and one law record `ohm` (a line fitted
to `2 x + 1`, record format 1, which carries no identifier) indexed by its law
identifier under it.

Read by `tests/previous_identifier_store.rs`: alice-zip 0.11 lists `135c407d…`
as a previous identifier, so every value reads back exact as
`ReadStatus::VerifiedPrevious`, and the law is found by that law identifier.
