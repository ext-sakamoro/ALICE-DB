"""Python binding tests — the source of truth for the README Python examples.

Run after `maturin develop --features python` (the python CI job does this).
Covers: the README quick-start example verbatim, the lossless vs lossy promise
for a gapped series, and that the wheel/module version is the crate version.
"""
import re
import tempfile
from pathlib import Path

import alice_db

_CARGO = Path(__file__).resolve().parents[2] / "Cargo.toml"


def _cargo_version() -> str:
    for line in _CARGO.read_text().splitlines():
        m = re.match(r'\s*version\s*=\s*"([^"]+)"', line)
        if m:
            return m.group(1)
    raise AssertionError("no version in Cargo.toml")


def test_readme_quickstart_example():
    # verbatim from README.md / README_JP.md (## Python); put_batch takes two
    # equal-length sequences, not a list of pairs
    with tempfile.TemporaryDirectory() as d:
        db = alice_db.open(str(Path(d) / "my_timeseries"))
        timestamps = list(range(1000))
        values = [i * 0.5 for i in range(1000)]
        db.put_batch(timestamps, values)
        got = db.get(100)
        avg = db.aggregate(0, 999, "avg")
        db.close()
    # the series is exactly linear (i*0.5), which the model fits exactly
    assert got is not None and abs(got - 50.0) < 1e-2, got
    assert abs(avg - 249.75) < 1.0, avg


def test_put_batch_rejects_mismatched_lengths():
    with tempfile.TemporaryDirectory() as d:
        db = alice_db.open(str(Path(d) / "mism"))
        try:
            db.put_batch([0, 1, 2], [1.0, 2.0])
            raise AssertionError("expected a ValueError for unequal lengths")
        except ValueError:
            pass
        finally:
            db.close()


def test_lossless_reads_a_gapped_series_back_exactly():
    # README "Irregular timestamps" + the Python lossless example: a gap (2 -> 10)
    ts = [0, 1, 2, 10]
    vals = [10.0, 20.0, 30.0, 31.0]
    with tempfile.TemporaryDirectory() as d:
        db = alice_db.open(str(Path(d) / "exact"), lossless=True)
        db.put_batch(ts, vals)
        db.flush()  # force the model store; lossless keeps residuals
        got = db.get(10)
        db.close()
    assert got is not None and got == 31.0, f"lossless get(10) = {got!r}, want 31.0"


def test_lossy_default_changes_a_gapped_point_after_flush():
    # the default (lossless=False) reads a gapped series on an even grid, so the
    # gapped point does not come back exactly — this is why lossless exists
    ts = [0, 1, 2, 10]
    vals = [10.0, 20.0, 30.0, 31.0]
    with tempfile.TemporaryDirectory() as d:
        db = alice_db.open(str(Path(d) / "lossy"))  # lossless=False default
        db.put_batch(ts, vals)
        db.flush()
        got = db.get(10)
        db.close()
    # not asserting the exact grid value (model-dependent), only that it changed
    assert got is None or got != 31.0, f"lossy get(10) = {got!r}, expected it to differ from 31.0"


def test_module_version_is_the_crate_version():
    # defect: wheel was pinned to 0.1.0 while the crate was 0.4.0; pyproject is now
    # dynamic so maturin takes the version from Cargo.toml — this guards the drift
    assert alice_db.__version__ == _cargo_version(), (alice_db.__version__, _cargo_version())
