# ALICE-DB

**Model-Based LSM-Tree Database** powered by [ALICE-Zip](https://github.com/ext-sakamoro/ALICE-Zip) procedural generation.

<p align="center">
  <img src="assets/concept.png" alt="ALICE-DB Concept" width="600">
</p>

## The Revolution: "Query the Function"

**Traditional databases store raw data:**
```
1000 sensor readings → 4KB on disk
Query: Read 4KB from disk → Decompress → Return
```

**ALICE-DB stores the mathematical model that generates the data:**
```
1000 sensor readings (linear trend) → "y = 0.5x + 10" → 16 bytes on disk
Query: Compute f(500) = 0.5 * 500 + 10 = 260.0 → Return
```

This is based on Kolmogorov complexity: *the shortest program that produces the output is the optimal representation.*

## Key Features

| Feature | Description |
|---------|-------------|
| **Extreme Compression** | 50-1000x for structured time-series data |
| **O(1) Point Queries** | Compute f(x) instead of disk read |
| **Automatic Model Selection** | Polynomial, Fourier, Perlin, LZMA fallback |
| **LSM-Tree Architecture** | Write-optimized with model-based SSTables |
| **Python + Rust** | High-level Python API, bare-metal Rust core |

## Quick Start

### Python

```bash
pip install alice-db
# or build from source:
cd ALICE-DB && pip install maturin && maturin develop --release
```

```python
import alice_db
import numpy as np

# Open database
db = alice_db.open("./my_timeseries")

# Insert time-series data
for i in range(10000):
    db.put(timestamp=i, value=np.sin(i * 0.01) * 100)

# Or batch insert with numpy (zero-copy)
timestamps = np.arange(10000, dtype=np.int64)
values = np.sin(timestamps * 0.01).astype(np.float32) * 100
db.put_numpy(timestamps, values)

# Query - this computes sin(5000 * 0.01) from the model, no disk read!
value = db.get(5000)

# Range query
points = db.scan(0, 9999)

# Aggregation
avg = db.aggregate(0, 9999, "avg")
total = db.aggregate(0, 9999, "sum")

# Downsampling (GROUP BY time interval)
hourly = db.downsample(0, 9999, interval=3600, agg="avg")

# Check compression stats
stats = db.stats()
print(f"Compression: {stats.average_compression_ratio:.1f}x")
print(f"Models used: {stats.model_distribution}")

db.close()
```

### Rust

```rust
use alice_db::{AliceDB, Aggregation};

fn main() -> std::io::Result<()> {
    let db = AliceDB::open("./my_timeseries")?;

    // Insert
    for i in 0..10000 {
        let value = (i as f32 * 0.01).sin() * 100.0;
        db.put(i, value)?;
    }

    // Query (computes from model!)
    if let Some(value) = db.get(5000)? {
        println!("Value at 5000: {}", value);
    }

    // Aggregation
    let avg = db.aggregate(0, 9999, Aggregation::Avg)?;
    println!("Average: {}", avg);

    // Stats
    let stats = db.stats();
    println!("Compression: {:.1}x", stats.average_compression_ratio);

    db.close()
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                       ALICE-DB                          │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌────────────┐  │
│  │  MemTable   │───▶│   Fitter    │───▶│  Segment   │  │
│  │ (BTreeMap)  │    │ Competition │    │  (Model)   │  │
│  └─────────────┘    └─────────────┘    └────────────┘  │
│         │                  │                  │        │
│         │         Polynomial, Fourier,        │        │
│         │         Sine, Perlin, LZMA          │        │
│         ▼                  ▼                  ▼        │
│  ┌─────────────────────────────────────────────────┐   │
│  │                 Storage Engine                   │   │
│  │  • Segment Index (time → model)                 │   │
│  │  • WAL (durability)                             │   │
│  │  • Compaction                                   │   │
│  └─────────────────────────────────────────────────┘   │
│                          │                             │
│                          ▼                             │
│  ┌─────────────────────────────────────────────────┐   │
│  │                   libalice                       │   │
│  │  • Polynomial fitting (Horner's method)         │   │
│  │  • Fourier analysis (FFT)                       │   │
│  │  • Perlin noise generation                      │   │
│  │  • LZMA compression (fallback)                  │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Model Types

ALICE-DB automatically selects the best model for your data:

| Model | Use Case | Compression |
|-------|----------|-------------|
| **Constant** | Flat lines | ∞ (8 bytes total) |
| **Linear** | Trends, ramps | ~250x |
| **Polynomial** | Curves, drift | ~50-200x |
| **Fourier** | Periodic signals | ~20-100x |
| **SineWave** | Simple oscillations | ~250x |
| **Perlin** | Noise patterns | ~40x |
| **RawLZMA** | Random data (fallback) | ~2-5x |

## Performance

### Write Performance

| Operation | Throughput |
|-----------|------------|
| Single insert | ~500K ops/sec |
| Batch insert | ~2M points/sec |
| Flush (model fitting) | ~1ms per 1000 points |

### Query Performance

| Operation | Latency |
|-----------|---------|
| Point query | **~120ns** (compute f(x)) |
| Range query (1000 points) | ~5µs |
| Aggregation (10K points) | ~50µs |

### Compression Ratios

| Data Type | Compression |
|-----------|-------------|
| Linear sensor data | **100-500x** |
| Sine wave (temperature) | **50-200x** |
| Polynomial trend | **50-150x** |
| Random noise | 2-5x (LZMA fallback) |

## Building from Source

### Requirements

- Rust 1.75+
- Python 3.9+ (for Python bindings)
- maturin (for Python package)

### Build

```bash
# Clone
git clone https://github.com/ext-sakamoro/ALICE-DB.git
cd ALICE-DB

# Build Rust library
cargo build --release

# Build Python package
pip install maturin
maturin develop --release

# Run tests
cargo test
pytest tests/

# Run benchmarks
cargo bench
```

## Related Projects

| Project | Description |
|---------|-------------|
| [ALICE-Zip](https://github.com/ext-sakamoro/ALICE-Zip) | Core procedural generation engine |
| [ALICE-Edge](https://github.com/ext-sakamoro/ALICE-Edge) | Embedded/IoT model generator (no_std) |
| [ALICE-Streaming-Protocol](https://github.com/ext-sakamoro/ALICE-Streaming-Protocol) | Ultra-low bandwidth video streaming |
| [ALICE-Eco-System](https://github.com/ext-sakamoro/ALICE-Eco-System) | Complete Edge-to-Cloud pipeline demo |

All projects share the core philosophy: **encode the generation process, not the data itself**.

## License

MIT License

## Author

Moroya Sakamoto

---

*"The best compression is to store the recipe, not the meal."*
