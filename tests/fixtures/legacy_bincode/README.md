`every_model.txt`: name and hex of `DataSegment::to_bytes` as written by alice-db at
`e5c0d81`, the last commit that wrote bincode, for the segments built in
`tests/legacy_every_model.rs`.

The `raw_*` lines were written again by `e5c0d81` with real LZMA payloads of nine samples (0..9 of
each type), because a reader now refuses a raw payload that does not decompress to `point_count`
samples; the first lines carried the placeholder payload `01 02 03`.
