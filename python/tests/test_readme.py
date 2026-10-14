"""Python binding tests — the source of truth for the README Python examples.

Run after `maturin develop --features python` (the python CI job does this).
The README's ```python blocks are EXTRACTED and executed here (not copied), so
the docs cannot drift from runnable code, and the lossless promise is asserted at
every written point (lossy and lossless agree at the endpoints, so a weaker test
would pass even if `lossless` were ignored).
"""
import re
import tempfile
from pathlib import Path

import alice_db

_ROOT = Path(__file__).resolve().parents[2]
_CARGO = _ROOT / "Cargo.toml"
_READMES = [_ROOT / "README.md", _ROOT / "README_JP.md"]


def _cargo_version() -> str:
    for line in _CARGO.read_text(encoding="utf-8").splitlines():
        m = re.match(r'\s*version\s*=\s*"([^"]+)"', line)
        if m:
            return m.group(1)
    raise AssertionError("no version in Cargo.toml")


def _python_blocks(md: str) -> list[str]:
    return re.findall(r"```python\n(.*?)```", md, re.DOTALL)


def test_readme_python_examples_execute():
    # run every ```python block of each README top to bottom in one namespace and a
    # scratch cwd, so a broken example (e.g. put_batch([(t, v)...])) fails here.
    # both README.md and README_JP.md are run (their code is identical, comments differ)
    import os
    for readme in _READMES:
        blocks = _python_blocks(readme.read_text(encoding="utf-8"))
        assert len(blocks) >= 2, f"{readme.name}: expected the Python examples, found {len(blocks)} blocks"
        ns: dict = {}
        with tempfile.TemporaryDirectory() as d:
            cwd = os.getcwd()
            os.chdir(d)
            try:
                for block in blocks:
                    exec(block, ns)  # noqa: S102 - executing our own README is the point
            finally:
                os.chdir(cwd)


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


def _read_back(path: Path, ts, vals, lossless: bool):
    db = alice_db.open(str(path), lossless=lossless)
    db.put_batch(ts, vals)
    db.flush()  # force the model store; lossless keeps residuals, lossy reads an even grid
    got = [db.get(t) for t in ts]
    db.close()
    return got


def test_lossless_reads_every_written_point_exactly():
    # the gap (2 -> 10) makes the lossy even-grid drift on the interior points; lossless
    # must return ALL written points, not just the endpoints (which both modes keep).
    ts = [0, 1, 2, 10]
    vals = [10.0, 20.0, 30.0, 31.0]
    with tempfile.TemporaryDirectory() as d:
        got = _read_back(Path(d) / "exact", ts, vals, lossless=True)
    assert got == vals, f"lossless read {got}, want {vals} (every written point must be exact)"


def test_lossy_default_drifts_on_a_gapped_interior_point():
    # this is WHY lossless exists: the default loses interior points of a gapped series.
    # it pins the lossy model's imprecision on THIS series; if the lossy fit ever improves
    # to exact here, this fails for a GOOD reason — the lossless test above is then no longer
    # a distinction, so move both to a harder gapped series rather than loosening this.
    ts = [0, 1, 2, 10]
    vals = [10.0, 20.0, 30.0, 31.0]
    with tempfile.TemporaryDirectory() as d:
        got = _read_back(Path(d) / "lossy", ts, vals, lossless=False)
    assert got != vals, (f"lossy read {got} == {vals}: a gapped interior point no longer drifts. "
                         "If the lossy fit genuinely improved, pick a harder gapped series so the "
                         "lossless test stays a real distinction (do not just loosen this assert).")


def test_module_version_is_the_crate_version():
    assert alice_db.__version__ == _cargo_version(), (alice_db.__version__, _cargo_version())
