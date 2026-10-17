A lossless store written by `alice-db` 0.3.0-beta.1 (`52ca813`, before residuals carried an
arithmetic identifier): 64 keys `0..=63`, value `2 sin(0.3 k + 0.4) + 0.25` (f32), one segment
`seg_1.rkyv` whose residual has no identifier (`ResidualSemantics::Unpinned`).

Read by `tests/residual_unpinned_store.rs`: the default read refuses it with
`ResidualError::Unpinned`; `ResidualPolicy::AcceptUnpinned` applies it and reports
`ReadStatus::Unpinned`.
