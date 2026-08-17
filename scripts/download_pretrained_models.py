#!/usr/bin/env python3
"""Download, verify, combine, and extract pretrained GitHub Release artifacts."""
import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts/manifest.json"
CACHE = ROOT / ".artifacts-cache"

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def artifact_files(item: dict) -> list[dict]:
    parts = item.get("parts")
    if parts:
        if not item.get("filename"):
            raise ValueError(f"Split artifact {item.get('name')} needs an assembled filename")
        return parts
    return [{key: item[key] for key in ("filename", "size", "sha256", "url")}]

def download_file(file_info: dict, force: bool) -> Path:
    url, filename = file_info.get("url", ""), file_info["filename"]
    if not url or "REPLACE_AFTER_RELEASE" in url:
        raise RuntimeError(f"Release URL has not been configured for {filename}")
    CACHE.mkdir(exist_ok=True)
    target, partial = CACHE / filename, CACHE / f"{filename}.part"
    if target.exists() and not force:
        if target.stat().st_size == file_info["size"] and sha256(target) == file_info["sha256"]:
            return target
        raise RuntimeError(f"Cached file failed validation: {target}; use --force to replace it")
    partial.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(url) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        if partial.stat().st_size != file_info["size"] or sha256(partial) != file_info["sha256"]:
            raise RuntimeError(f"Size or SHA256 verification failed for {filename}")
        partial.replace(target)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return target

def assemble(item: dict, downloaded: list[Path], force: bool) -> Path:
    if len(downloaded) == 1 and not item.get("parts"):
        return downloaded[0]
    target = CACHE / item["filename"]
    if target.exists() and not force:
        if target.stat().st_size == item["size"] and sha256(target) == item["sha256"]:
            return target
        raise RuntimeError(f"Assembled artifact failed validation: {target}; use --force to replace it")
    partial = CACHE / f"{item['filename']}.assembling"
    partial.unlink(missing_ok=True)
    try:
        with partial.open("wb") as output:
            for path in downloaded:
                with path.open("rb") as source:
                    shutil.copyfileobj(source, output)
        if partial.stat().st_size != item["size"] or sha256(partial) != item["sha256"]:
            raise RuntimeError(f"Combined size or SHA256 verification failed for {item['filename']}")
        partial.replace(target)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return target

def safe_extract(archive: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        root = ROOT.resolve()
        for member in tar.getmembers():
            destination = (ROOT / member.name).resolve()
            if root not in destination.parents and destination != root:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"Archive links are not allowed: {member.name}")
        tar.extractall(ROOT, filter="data")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    mode = "inference" if args.inference_only else "full"
    from verify_artifacts import validate
    if args.verify_only:
        errors = validate(mode=mode, load_models=True)
        if errors:
            raise SystemExit("Artifact validation failed:\n  - " + "\n  - ".join(errors))
        print(f"All {mode} pretrained artifacts are complete and valid.")
        return
    manifest = json.loads(MANIFEST.read_text())
    items = [x for x in manifest["artifacts"] if not (args.inference_only and x.get("oof_only"))]
    for item in items:
        files = [download_file(part, args.force) for part in artifact_files(item)]
        safe_extract(assemble(item, files, args.force))
    errors = validate(mode=mode, load_models=True)
    if errors:
        raise SystemExit("Post-download validation failed:\n  - " + "\n  - ".join(errors))
    print(f"Pretrained {mode} artifacts downloaded, verified, and extracted.")

if __name__ == "__main__":
    main()
