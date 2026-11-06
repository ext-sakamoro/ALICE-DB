Stores written by `alice-db` at `e5c0d81`, the last commit that named a store's
files in two ways. Written by a program that opens `BlobStorage` with these
arguments; the flush mode comes first.

- `overwrite`: `x.wal`, overwrite, put k v, flush, put w u (`x.sst`, which that
  build wrote and did not read)
- `append`: `x.wal`, append, put k v, flush, put w u (`blob-000001.sst`)
- `two_overwrite`: `x.wal`, overwrite, put kx vx, flush, put wx ux; then `y.wal`
  the same with ky, wy
- `x_and_blob_append`: `x.wal`, append, put kx vx, flush; then `blob.wal`, append,
  put kb vb, flush (both numbered files hold one store's data each)
- `blob_wal`: `blob.wal`, append, put k v, flush, put w u

Read by `tests/blob_one_store.rs`.
