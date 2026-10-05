#!/usr/bin/env python3
"""Private donor→reader stream for six 0600 files; never widens permissions."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import pwd
import stat
import sys
import tarfile

from grant_migration_read import FILES, SOURCE


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def donor_manifest():
    if pwd.getpwuid(os.geteuid()).pw_name != "anhntc2" or SOURCE.resolve() != SOURCE:
        raise RuntimeError("Exact donor account/source required")
    rows = {}
    for name in FILES:
        path = SOURCE / name
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
            raise RuntimeError("Reviewed protected source scope changed")
        rows[name] = {"sha256": digest(path), "size": info.st_size, "mode": 0o600, "mtime_ns": info.st_mtime_ns}
    if sum(row["size"] for row in rows.values()) > 16 * 1024 * 1024:
        raise RuntimeError("Donor metadata stream exceeds the reviewed small-file budget")
    return {"source_root": str(SOURCE), "files": rows, "permissions_modified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--emit-tar", action="store_true")
    group.add_argument("--manifest-only", action="store_true")
    group.add_argument("--receive-dir", type=Path)
    args = parser.parse_args()
    if args.receive_dir:
        if (pwd.getpwuid(os.geteuid()).pw_name != "hoangnv242" or args.receive_dir.exists()
                or args.receive_dir.resolve() != Path("/mnt/data/users/hoangnv242/memrec-migration-20261005-hnv/protected-source-hnv")
                or os.environ.get("CUDA_VISIBLE_DEVICES") != "" or not os.environ.get("SLURM_JOB_ID", "").isdigit()):
            raise RuntimeError("Private staging must be new and inside the approved reader's CPU step")
        args.receive_dir.mkdir(mode=0o700)
        allowed, seen, total = set(FILES) | {"donor-manifest.json"}, set(), 0
        with tarfile.open(fileobj=sys.stdin.buffer, mode="r|") as archive:
            for member in archive:
                if member.name not in allowed or member.name in seen or not member.isfile():
                    raise RuntimeError("Donor archive member outside the exact allowlist")
                total += member.size
                if total > 16 * 1024 * 1024:
                    raise RuntimeError("Donor archive budget exceeded")
                path = args.receive_dir / member.name
                path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as output:
                    output.write(archive.extractfile(member).read())
                seen.add(member.name)
        if seen != allowed:
            raise RuntimeError("Incomplete protected-file stream")
        manifest = json.loads((args.receive_dir / "donor-manifest.json").read_text())
        if manifest["source_root"] != str(SOURCE) or set(manifest["files"]) != set(FILES):
            raise RuntimeError("Wrong protected-file donor manifest")
        for name, row in manifest["files"].items():
            path = args.receive_dir / name
            if path.stat().st_size != row["size"] or digest(path) != row["sha256"]:
                raise RuntimeError("Protected file hash mismatch")
        print(json.dumps({"status": "PRIVATE_DONOR_SNAPSHOT_VERIFIED", "files": len(FILES), "bytes": total}), flush=True)
    else:
        manifest = donor_manifest()
        if args.manifest_only:
            print(json.dumps(manifest, sort_keys=True))
            return
        with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
            for name in FILES:
                archive.add(SOURCE / name, arcname=name, recursive=False)
            payload = json.dumps(manifest, sort_keys=True).encode()
            member = tarfile.TarInfo("donor-manifest.json")
            member.size, member.mode = len(payload), 0o600
            archive.addfile(member, io.BytesIO(payload))


if __name__ == "__main__":
    main()
