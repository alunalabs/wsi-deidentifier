#!/usr/bin/env python3
"""
Re-identifies de-identified Whole Slide Image (WSI) files by resolving salted hashes back to original filenames.

This script helps reverse the de-identification process by mapping salted hashes back to their
original slide identifiers. It can work with either individual hashes or process entire
deidentified directories to find matching original slides.

Usage Examples
-------------
# Re-identify a single hash
uv run python reidentify.py --hash "abc123def456..." --identified "sample/identified/" --salt "your-secret-salt"

# Re-identify all slides in a deidentified directory
uv run python reidentify.py --deidentified "sample/deidentified/" --identified "sample/identified/" --salt "your-secret-salt"

# Re-identify a specific hash (positional argument for backwards compatibility)
uv run python reidentify.py abc123def456... --identified "sample/identified/" --salt "your-secret-salt"

The script will:
1. Build an index of all .svs files in the identified directory using salted hashes
2. Match the provided hash(es) against the index
3. Return the original file path(s) for successful matches
"""

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


def process_deidentified_dir(
    deidentified_dir: Path, identified_dir: Path, salt: str
) -> None:
    """Process all files in a deidentified directory and find their original matches."""
    index = build_index(identified_dir, salt)
    deidentified_files = list(deidentified_dir.rglob("*.svs"))

    # Filter out macOS resource fork files (._filename)
    deidentified_files = [f for f in deidentified_files if not f.name.startswith("._")]

    if not deidentified_files:
        print("No .svs files found in deidentified directory.", file=sys.stderr)
        return

    matches_found = 0
    for deidentified_file in deidentified_files:
        # The deidentified filename should be the hash
        hash_name = deidentified_file.stem
        match = index.get(hash_name.lower())

        if match:
            print(f"Hash: {hash_name} -> Original: {match}")
            matches_found += 1
        else:
            print(f"Hash: {hash_name} -> No match found", file=sys.stderr)

    print(f"\nFound {matches_found} matches out of {len(deidentified_files)} files.")


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Re-identify a salted-hash slide ID or directory of deidentified slides"
    )

    # Create a mutually exclusive group for hash vs deidentified directory
    input_group = p.add_mutually_exclusive_group()
    input_group.add_argument("--hash", help="salted SHA-256 hash you want to resolve")
    input_group.add_argument(
        "--deidentified",
        type=Path,
        help="directory containing deidentified .svs slides to re-identify",
    )

    # For backwards compatibility, also accept hash as positional argument
    p.add_argument(
        "hash_positional",
        nargs="?",
        help="salted SHA-256 hash (positional argument for backwards compatibility)",
    )

    p.add_argument(
        "-i",
        "--identified",
        type=Path,
        required=True,
        help="directory with the original (identified) .svs slides",
    )
    p.add_argument("--salt", required=True, help="the same secret salt")
    args = p.parse_args(argv)

    # Determine which hash to use (prioritize --hash, then positional)
    target_hash = args.hash or args.hash_positional

    # Ensure we have either a hash or deidentified directory
    if not target_hash and not args.deidentified:
        p.error(
            "Must provide either --hash, --deidentified directory, or hash as positional argument"
        )

    # Ensure we don't have both hash and deidentified directory
    if target_hash and args.deidentified:
        p.error("Cannot specify both hash and deidentified directory")

    if args.deidentified:
        # Process entire deidentified directory
        process_deidentified_dir(args.deidentified, args.identified, args.salt)
    else:
        # Process single hash
        index = build_index(args.identified, args.salt)
        match = index.get(target_hash.lower())

        if match:
            print(f"Match: {match}")
        else:
            print("No slide with that hash found.", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
