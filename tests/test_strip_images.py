import subprocess
import sys
from pathlib import Path

import pytest

try:
    import numpy as np
    import tifffile
    import tiffparser
except Exception:
    pytest.skip("tifffile or tiffparser not available", allow_module_level=True)


def create_slide(path: Path) -> None:
    """Create a simple TIFF slide with label and macro images."""
    main = np.zeros((10, 10, 3), dtype=np.uint8)
    label = np.ones((4, 4, 3), dtype=np.uint8)
    macro = np.full((6, 6, 3), 128, dtype=np.uint8)
    with tifffile.TiffWriter(str(path)) as tif:
        tif.write(main, photometric="rgb", description="Aperio Image Library")
        tif.write(label, photometric="rgb", description="label")
        tif.write(macro, photometric="rgb", description="macro")


def descriptions(path: Path):
    with open(path, "rb") as fh:
        t = tiffparser.TiffFile(fh)
        descs = []
        for page in t.pages:
            tag = page.tags.get("ImageDescription")
            if not tag:
                descs.append("")
            else:
                val = tag.value
                if isinstance(val, bytes):
                    val = val.decode(errors="ignore")
                descs.append(val)
        return descs


def test_strip_all_images(tmp_path):
    slide = tmp_path / "slide.tiff"
    create_slide(slide)

    # Ensure label and macro exist initially
    descs = descriptions(slide)
    assert any("label" in d for d in descs)
    assert any("macro" in d for d in descs)

    out_dir = tmp_path / "out"
    map_csv = tmp_path / "map.csv"
    cmd = [sys.executable, "deidentify.py", str(slide), "-o", str(out_dir), "-m", str(map_csv), "--strip-all-images"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    out_file = out_dir / slide.name
    descs_after = descriptions(out_file)
    assert not any("label" in d for d in descs_after)
    assert not any("macro" in d for d in descs_after)
    assert len(descs_after) == 1
