"""Verify release artifacts without executing or unpickling a checkpoint."""

import argparse
import hashlib
from pathlib import Path

EXPECTED = {
    "head": "a7f5dbda0843fe2bbf562f0a2daea5eeb4969290e2867ee1eeb0c5681d0fed60",
    "backbone": "c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", required=True, type=Path)
    parser.add_argument("--backbone", required=True, type=Path)
    args = parser.parse_args()
    for name, expected in EXPECTED.items():
        h = hashlib.sha256()
        with getattr(args, name).open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(block)
        if h.hexdigest() != expected:
            raise SystemExit(f"{name}: checksum mismatch")
        print(f"{name}: verified")


if __name__ == "__main__":
    main()
