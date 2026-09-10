"""Download official source files listed in data_sources/source_registry.csv.

This script is intentionally small and conservative:
- it only downloads rows with a non-empty download_url
- it never overwrites an existing raw file unless --overwrite is passed
- it writes a manifest with timestamps, byte counts, and SHA256 hashes
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import ssl
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = (ROOT / "data_raw").resolve()
REGISTRY = ROOT / "data_sources" / "source_registry.csv"
MANIFEST = ROOT / "data_sources" / "download_manifest.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = Path(unquote(parsed.path)).name
    if not name:
        raise ValueError(f"Could not infer filename from URL: {url}")
    return name


def destination_for(raw_subdir: str, url: str) -> Path:
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https":
        raise ValueError(f"Only HTTPS downloads are permitted: {url}")
    dest = (ROOT / raw_subdir / filename_from_url(url)).resolve()
    if not dest.is_relative_to(RAW_ROOT):
        raise ValueError(f"Download destination must remain under data_raw: {raw_subdir}")
    return dest


def load_registry(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def download(url: str, dest: Path, timeout: int, overwrite: bool) -> dict[str, object]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        return {
            "status": "skipped_exists",
            "path": str(dest.relative_to(ROOT)),
            "bytes": dest.stat().st_size,
            "sha256": sha256_file(dest),
        }

    request = Request(
        url,
        headers={
            "User-Agent": (
                "AustralianTransitionRiskModel/0.1 "
                "(research data provenance; contact: local project)"
            )
        },
    )
    context = ssl.create_default_context()
    with urlopen(request, timeout=timeout, context=context) as response:
        data = response.read()
    dest.write_bytes(data)
    return {
        "status": "downloaded",
        "path": str(dest.relative_to(ROOT)),
        "bytes": len(data),
        "sha256": sha256_file(dest),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--source-id", action="append", default=[])
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = load_registry(args.registry)
    selected = set(args.source_id)
    if selected:
        rows = [row for row in rows if row["source_id"] in selected]

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with MANIFEST.open("a", encoding="utf-8") as manifest:
        for row in rows:
            url = row.get("download_url", "").strip()
            if not url:
                continue

            raw_subdir = row.get("raw_subdir", "").strip()
            if not raw_subdir:
                print(f"Skipping {row['source_id']}: no raw_subdir", file=sys.stderr)
                continue

            dest = destination_for(raw_subdir, url)
            record = {
                "timestamp_utc": utc_now(),
                "source_id": row["source_id"],
                "source_name": row["source_name"],
                "url": url,
            }
            try:
                record.update(download(url, dest, args.timeout, args.overwrite))
                print(f"{record['status']}: {row['source_id']} -> {record['path']}")
            except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
                failures += 1
                record.update({"status": "failed", "error": repr(exc)})
                print(f"failed: {row['source_id']}: {exc}", file=sys.stderr)
            manifest.write(json.dumps(record, ensure_ascii=True) + "\n")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
