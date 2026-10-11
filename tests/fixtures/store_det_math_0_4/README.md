A lossless store written by `alice-db` 0.3.0 (`alice-det-math` 0.4, arithmetic
identifier `d2209b30…`): 64 keys `0..=63`, value `2 sin(0.3 k + 0.4) + 0.25` (f32),
one segment `seg_1.rkyv` whose residual is pinned to that identifier.

Read by `tests/residual_arithmetic.rs`: a build that computes with another
arithmetic must refuse its exact values rather than return the model.
