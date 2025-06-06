import urllib.request
from pathlib import Path

import pytest

import replace_macro

pytest.importorskip("openslide")

SAMPLE_URL = "https://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/CMU-1-Small-Region.svs"


def fetch_sample(dest: Path) -> None:
    if not dest.exists():
        with urllib.request.urlopen(SAMPLE_URL) as resp, open(dest, "wb") as fh:
            fh.write(resp.read())


def test_load_macro_uses_openslide(tmp_path):
    slide_path = tmp_path / "sample.svs"
    fetch_sample(slide_path)

    info = replace_macro.read_svs_macro_info(str(slide_path))

    img_generic = replace_macro.load_macro(str(slide_path), info)
    img_os = replace_macro.load_macro_openslide(str(slide_path))

    assert img_os is not None
    assert img_generic is not None
    assert img_generic.size == img_os.size
    assert img_generic.tobytes() == img_os.tobytes()
