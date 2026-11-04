"""Download the five required TNG100-1 group-catalog fields, one request each.

Requires TNG_API_KEY in the environment. The key is used only for requests,
never persisted in URLs, logs, or provenance. No account creation is attempted.
"""
from __future__ import annotations

import argparse
import getpass
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.error
import urllib.parse
import urllib.request


SIMULATION = "TNG100-1"
BASE_URL = "https://www.tng-project.org/api/"
FIELDS = ("SubhaloPos", "SubhaloMassType", "SubhaloLenType", "SubhaloFlag", "SubhaloGrNr")
SNAPSHOTS = (99, 67, 50, 33)


class DownloadError(RuntimeError):
    """A sanitized error safe to display without authentication material."""


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlparse(newurl)
        host = parsed.hostname or ""
        if parsed.scheme != "https":
            raise DownloadError("Refused a download redirect without HTTPS.")
        if not (host == "tng-project.org" or host.endswith(".tng-project.org")):
            raise DownloadError("Refused redirect outside the official TNG domain.")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected and urllib.parse.urlparse(req.full_url).hostname != host:
            # Raw downloads use signed redirects. Do not forward the account key
            # to a different host; its temporary download token remains in memory.
            redirected.remove_header("Api-key")
        return redirected


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(obj, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def open_authenticated(url: str, key: str, timeout: float):
    request = urllib.request.Request(url, headers={"api-key": key, "User-Agent": "GalaxySpatialAnalytics/1.0"})
    opener = urllib.request.build_opener(SafeRedirectHandler())
    try:
        return opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        # Do not interpolate exc, URL, response body, or redirect query tokens.
        raise DownloadError(f"TNG returned HTTP {exc.code}; check account/API access.") from None
    except urllib.error.URLError:
        raise DownloadError("TNG network request failed; check connectivity and retry.") from None


def get_metadata(snapshot: int, key: str, timeout: float) -> dict:
    url = f"{BASE_URL}{SIMULATION}/snapshots/{snapshot}/"
    with open_authenticated(url, key, timeout) as response:
        try:
            payload = json.load(response)
        except (ValueError, UnicodeDecodeError):
            raise DownloadError("Snapshot metadata response was not valid JSON.") from None
    if int(payload.get("number", -1)) != snapshot:
        raise DownloadError("Snapshot metadata does not match the requested snapshot.")
    return {
        "simulation": SIMULATION,
        "snapshot": snapshot,
        "redshift": float(payload["redshift"]),
        "source_url": url,
        "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "boxsize_cMpc_h": 75.0,
        "little_h": 0.6774,
        "geometry_source": "https://www.tng-project.org/data/downloads/TNG100-1/",
        "geometry_note": "75 cMpc/h full periodic cube; verify against file metadata where available.",
        "num_subhalos": payload.get("num_groups_subfind"),
    }


def download_field(data_dir: Path, snapshot: int, field: str, key: str,
                   timeout: float = 180.0, force: bool = False) -> dict:
    directory = data_dir / SIMULATION / f"snapshot_{snapshot:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{field}.hdf5"
    manifest_path = directory / f"{field}.provenance.json"
    url = f"{BASE_URL}{SIMULATION}/files/groupcat-{snapshot}/?Subhalo={field}"
    if destination.exists() or manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            intact = (destination.is_file() and manifest.get("simulation") == SIMULATION
                      and manifest.get("snapshot") == snapshot and manifest.get("field") == field
                      and manifest["source_url"] == url
                      and manifest["sha256"] == sha256_file(destination)
                      and manifest["measured_bytes"] == destination.stat().st_size)
        except (OSError, ValueError, KeyError):
            intact = False
        if intact and not force:
            print(f"Verified existing field: snapshot {snapshot}, {field}", flush=True)
            return manifest
        if not force:
            raise DownloadError(f"Existing snapshot {snapshot}/{field} has missing or differing provenance; use --force explicitly.")
    descriptor, temporary = tempfile.mkstemp(prefix=field + ".", suffix=".part", dir=directory)
    digest = hashlib.sha256()
    measured = 0
    try:
        with os.fdopen(descriptor, "wb") as stream:
            with open_authenticated(url, key, timeout) as response:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
                    digest.update(block)
                    measured += len(block)
        with open(temporary, "rb") as stream:
            if stream.read(8) != b"\x89HDF\r\n\x1a\n":
                raise DownloadError(f"Snapshot {snapshot}/{field} response was not an HDF5 file.")
        manifest = {
            "simulation": SIMULATION, "snapshot": snapshot, "field": field,
            "source_url": url, "measured_bytes": measured, "sha256": digest.hexdigest(),
            "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "authentication": "TNG_API_KEY environment variable; key not persisted",
        }
        os.replace(temporary, destination)
        atomic_json(manifest_path, manifest)
        print(f"Downloaded snapshot {snapshot}, {field}: {measured / 1e6:.1f} MB", flush=True)
        return manifest
    except urllib.error.HTTPError as exc:
        raise DownloadError(f"TNG streaming request returned HTTP {exc.code}.") from None
    except (urllib.error.URLError, TimeoutError):
        raise DownloadError("TNG streaming request failed; partial download was discarded.") from None
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--snapshots", type=int, nargs="+", default=list(SNAPSHOTS))
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--force", action="store_true", help="Explicitly permit replacing existing data/provenance")
    parser.add_argument("--prompt-key", action="store_true", help="Read API key privately in a local terminal (not stored)")
    args = parser.parse_args()
    key = os.environ.get("TNG_API_KEY", "").strip()
    if not key and args.prompt_key:
        key = getpass.getpass("TNG API key (hidden; used only in this process): ").strip()
    if not key:
        print("No download attempted: configure TNG_API_KEY or provide existing field files.")
        return 2
    if not set(args.snapshots).issubset(SNAPSHOTS):
        parser.error("Supported experiment snapshots are 99, 67, 50, and 33.")
    try:
        for snapshot in args.snapshots:
            metadata_path = args.data_dir / SIMULATION / f"snapshot_{snapshot:03d}" / "metadata.json"
            if metadata_path.exists() and not args.force:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("snapshot") != snapshot or metadata.get("simulation") != SIMULATION:
                    raise DownloadError("Existing snapshot metadata mismatch; use --force explicitly.")
            else:
                atomic_json(metadata_path, get_metadata(snapshot, key, args.timeout))
            for field in FIELDS:
                download_field(args.data_dir, snapshot, field, key, args.timeout, args.force)
    except (DownloadError, OSError, ValueError, KeyError) as exc:
        if isinstance(exc, DownloadError):
            print(f"Download stopped: {exc}")
        else:
            print("Download stopped: file or metadata validation error; inspect local paths and retry.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
