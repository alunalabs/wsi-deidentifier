import csv
import subprocess
import sys
from pathlib import Path

import pytest

try:
    from PIL import Image
except Exception:
    pytest.skip("Pillow not available", allow_module_level=True)

try:
    import matplotlib.pyplot as plt  # noqa: F401
except Exception:
    pytest.skip("matplotlib not available", allow_module_level=True)


def create_simple_slide(path: Path) -> None:
    """Create a minimal TIFF slide for testing."""
    img = Image.new("RGB", (10, 10), color="white")
    img.save(path)


def test_deidentify_cli(tmp_path):
    slide = tmp_path / "simple.tiff"
    create_simple_slide(slide)
    out_dir = tmp_path / 'out'
    map_csv = tmp_path / 'map.csv'
    cmd = [sys.executable, 'deidentify.py', str(slide), '-o', str(out_dir), '-m', str(map_csv)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    out_files = list(out_dir.glob('*'))
    assert out_files, 'no output files generated'
    assert map_csv.exists(), 'mapping csv not created'
    with open(map_csv) as f:
        rows = list(csv.reader(f))
    assert len(rows) == 2  # header + one entry
