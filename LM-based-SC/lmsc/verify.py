"""Verify frozen research assets without loading pickles or requiring ML packages."""
import argparse
import hashlib
import json
from .paths import DATA, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    manifest = json.loads((DATA / "assets.json").read_text())
    failures = 0
    for item in manifest["assets"]:
        path = ROOT / item["path"]
        if not path.is_file():
            print(f"MISSING {item['path']}")
            failures += 1
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        valid = digest.hexdigest() == item["sha256"] and path.stat().st_size == item["bytes"]
        print(f"{'OK' if valid else 'CHANGED'} {item['path']}")
        failures += not valid
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
