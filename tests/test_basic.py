"""
Basic tests for ALICE-DB Python bindings.

Run with: pytest tests/test_basic.py -v
"""

import tempfile
import os
import pytest

# Skip all tests if extension not built
pytest.importorskip("alice_db.alice_db")

import alice_db
import numpy as np


class TestBasicOperations:
    """Test basic database operations."""

    def test_open_close(self):
        """Test database open and close."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = alice_db.open(tmpdir)
            assert db is not None
            db.close()

    def test_context_manager(self):
        """Test context manager protocol."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with alice_db.open(tmpdir) as db:
                db.put(0, 1.0)
                db.flush()

    def test_put_get(self):
        """Test single insert and query."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with alice_db.open(tmpdir) as db:
                db.put(100, 42.0)
                db.flush()

                # Note: May not find exact point due to segment storage
                # but should work after flush
                points = db.scan(0, 200)
                assert len(points) > 0

    def test_put_batch(self):
        """Test batch insert."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with alice_db.open(tmpdir) as db:
                timestamps = list(range(100))
                values = [float(i) for i in range(100)]
                db.put_batch(timestamps, values)
                db.flush()

                stats = db.stats()
                assert stats.total_segments >= 1


class TestNumpyIntegration:
    """Test numpy array integration."""

    def test_put_numpy(self):
        """Test numpy array insert."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with alice_db.open(tmpdir) as db:
                timestamps = np.arange(100, dtype=np.int64)
                values = np.arange(100, dtype=np.float32) * 0.5
                db.put_numpy(timestamps, values)
                db.flush()

                stats = db.stats()
                assert stats.total_segments >= 1

    def test_scan_numpy(self):
        """Test numpy array query."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with alice_db.open(tmpdir) as db:
                # Insert
                timestamps = np.arange(100, dtype=np.int64)
                values = np.arange(100, dtype=np.float32)
                db.put_numpy(timestamps, values)
                db.flush()

                # Query
                result = db.scan_numpy(0, 99)
                assert result.shape[1] == 2  # (N, 2) array
                assert len(result) > 0


class TestAggregations:
    """Test aggregation queries."""

    def test_sum(self):
        """Test sum aggregation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with alice_db.open(tmpdir) as db:
                for i in range(100):
                    db.put(i, float(i))
                db.flush()

                total = db.aggregate(0, 99, "sum")
                expected = sum(range(100))  # 0+1+...+99 = 4950
                assert abs(total - expected) < expected * 0.1  # Within 10%

    def test_avg(self):
        """Test average aggregation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with alice_db.open(tmpdir) as db:
                for i in range(100):
                    db.put(i, float(i))
                db.flush()

                avg = db.aggregate(0, 99, "avg")
                assert avg > 40 and avg < 60  # Should be ~49.5

    def test_min_max(self):
        """Test min/max aggregation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with alice_db.open(tmpdir) as db:
                for i in range(100):
                    db.put(i, float(i))
                db.flush()

                min_val = db.aggregate(0, 99, "min")
                max_val = db.aggregate(0, 99, "max")

                assert min_val < 10  # Should be close to 0
                assert max_val > 90  # Should be close to 99


class TestDownsampling:
    """Test downsampling queries."""

    def test_downsample_avg(self):
        """Test downsampling with average."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with alice_db.open(tmpdir) as db:
                for i in range(100):
                    db.put(i, float(i))
                db.flush()

                # Downsample to 10 buckets
                result = db.downsample(0, 99, 10, "avg")
                assert len(result) <= 10


class TestCompression:
    """Test compression effectiveness."""

    def test_linear_compression(self):
        """Test that linear data compresses well."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with alice_db.open(tmpdir, memtable_capacity=1000) as db:
                # Insert perfectly linear data
                for i in range(1000):
                    db.put(i, float(i) * 2.0 + 10.0)
                db.flush()

                stats = db.stats()
                # Linear data should compress extremely well
                print(f"Compression ratio: {stats.average_compression_ratio:.1f}x")
                print(f"Model distribution: {stats.model_distribution}")

                assert stats.average_compression_ratio > 10  # At least 10x

    def test_sine_wave_compression(self):
        """Test that sine wave data compresses well."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with alice_db.open(tmpdir, memtable_capacity=1000) as db:
                # Insert sine wave
                for i in range(1000):
                    value = np.sin(i * 0.01 * 2 * np.pi)
                    db.put(i, float(value))
                db.flush()

                stats = db.stats()
                print(f"Compression ratio: {stats.average_compression_ratio:.1f}x")
                print(f"Model distribution: {stats.model_distribution}")

                # Should achieve good compression via Fourier/SineWave
                assert stats.average_compression_ratio > 5


class TestStats:
    """Test statistics."""

    def test_stats_repr(self):
        """Test stats representation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with alice_db.open(tmpdir) as db:
                for i in range(100):
                    db.put(i, float(i))
                db.flush()

                stats = db.stats()
                repr_str = repr(stats)
                assert "Stats" in repr_str
                assert "segments" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
