#!/usr/bin/env python3
import argparse
import hashlib
import sys
from pathlib import Path


def salted_hash(text: str, salt: str) -> str:
    return hashlib.sha256((salt + text).encode()).hexdigest()


def build_index(identified_dir: Path, salt: str) -> dict[str, Path]:
    """
    Return {hash -> original_path} for every .svs file under identified_dir.
    """
    index = {}
    for slide in identified_dir.rglob("*.svs"):
        slide_id = slide.stem  # e.g. "CMU-1"
        h = salted_hash(slide_id, salt)
        index[h] = slide
    return index


def main(argv=None):
    p = argparse.ArgumentParser(description="Re-identify a salted-hash slide ID")
    p.add_argument("hash", help="salted SHA-256 you want to resolve")
    p.add_argument(
        "-i",
        "--identified",
        required=True,
        help="directory with the original (identified) .svs slides",
    )
    p.add_argument("--salt", required=True, help="the same secret salt")
    args = p.parse_args(argv)

    index = build_index(Path(args.identified), args.salt)
    match = index.get(args.hash.lower())

    if match:
        print(f"Match: {match}")
    else:
        print("No slide with that hash found.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
