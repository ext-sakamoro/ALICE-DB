"""Writes residual_alone_dict1g.lzma for tests/bounded_residual_memory.rs: 64 MiB
of zeros as an LZMA "alone" stream (liblzma) whose header declares a 1 GiB
dictionary, which a decoder must not reserve.

Run from the repository root:

    python tests/data/make_residual_bomb.py
"""

import lzma
import pathlib
import struct

OUT = pathlib.Path(__file__).resolve().parent / "residual_alone_dict1g.lzma"

s = bytearray(lzma.compress(bytes(64 * 1024 * 1024), format=lzma.FORMAT_ALONE))
s[1:5] = struct.pack("<I", 1 << 30)
OUT.write_bytes(bytes(s))
print(f"{OUT.name}: {len(s)} bytes")
