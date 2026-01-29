"""
ALICE-DB: Model-Based LSM-Tree Database

A revolutionary database that stores mathematical models instead of raw data,
achieving extreme compression ratios for time-series and numerical data.

Example:
    >>> import alice_db
    >>>
    >>> # Open database
    >>> db = alice_db.open("./my_data")
    >>>
    >>> # Insert data
    >>> db.put(timestamp=100, value=42.0)
    >>> db.put_batch([0, 1, 2, 3], [0.0, 1.0, 2.0, 3.0])
    >>>
    >>> # Query
    >>> value = db.get(100)
    >>> points = db.scan(0, 999)
    >>> avg = db.aggregate(0, 999, "avg")
    >>>
    >>> # Flush and close
    >>> db.flush()
    >>> db.close()

For numpy integration:
    >>> import numpy as np
    >>> timestamps = np.arange(1000, dtype=np.int64)
    >>> values = np.sin(timestamps * 0.01).astype(np.float32)
    >>> db.put_numpy(timestamps, values)
    >>> result = db.scan_numpy(0, 999)  # Returns (N, 2) array

Author: Moroya Sakamoto
License: MIT
"""

# Import from Rust extension
try:
    from alice_db.alice_db import (
        AliceDB,
        Stats,
        open,
        __version__,
        __author__,
    )
except ImportError:
    # Fallback for development/documentation
    __version__ = "0.1.0"
    __author__ = "Moroya Sakamoto"

    def open(path: str, memtable_capacity: int = 1000, enable_wal: bool = True):
        """Open a database at the specified path."""
        raise ImportError("alice_db Rust extension not built. Run: maturin develop --release")

__all__ = [
    "AliceDB",
    "Stats",
    "open",
    "__version__",
    "__author__",
]
