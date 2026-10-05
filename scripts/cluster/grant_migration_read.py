#!/usr/bin/env python3
"""Grant named-user READ on six reviewed source files, never world/group read.

Small metadata-only donor-account operation. No compute, model or job control.
Reject existing extended ACLs rather than expanding masked rights of others.
"""

import errno
import os
from pathlib import Path
import pwd
import stat
import struct

SOURCE = Path("/mnt/data/users/anhnct/memrec-hnv")
FILES = (
    "cache/pip/http/8/9/1/7/2/89172e1b67e9e3c9854025385bfb7d37adaa89601789d98df17ff2ea",
    "cache/pip/http/a/1/9/5/3/a19537d3cf37c122db841d6fe4cd322bc10d1a558bb00d146b85cb9a",
    "models/Qwen3.5-4B-851bf6e-hnv/.cache/huggingface/trees/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a.json",
    "models/all-MiniLM-L6-v2-1110a24-hnv/.cache/huggingface/trees/1110a243fdf4706b3f48f1d95db1a4f5529b4d41.json",
    "repo/MemRec-hnv/data/amazon_books/artifacts/p6_prepared-hnv.json",
    "repo/MemRec-hnv/data/amazon_books/artifacts/p7_prepared-hnv.json",
)


def read_acl(reader_uid: int) -> bytes:
    # Linux POSIX ACL xattr: version2, owner rw, this named user r, original
    # owning group none, mask r, everyone else none. No write permission added.
    entries = ((1, 6, 0xFFFFFFFF), (2, 4, reader_uid), (4, 0, 0xFFFFFFFF),
               (16, 4, 0xFFFFFFFF), (32, 0, 0xFFFFFFFF))
    return struct.pack("<I", 2) + b"".join(struct.pack("<HHI", *entry) for entry in entries)


def main():
    donor = pwd.getpwnam("anhntc2").pw_uid
    reader = pwd.getpwnam("hoangnv242").pw_uid
    if os.geteuid() != donor or SOURCE.is_symlink() or SOURCE.resolve() != SOURCE:
        raise RuntimeError("Exact donor owner/source required")
    # Validate ALL targets before any write; never overwrite an unknown ACL.
    for relative in FILES:
        path = SOURCE / relative
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != donor or stat.S_IMODE(info.st_mode) != 0o600:
            raise RuntimeError("Reviewed donor-only file scope changed")
        try:
            os.getxattr(path, "system.posix_acl_access")
        except OSError as error:
            if error.errno != errno.ENODATA:
                raise
        else:
            raise RuntimeError("Existing extended ACL needs explicit review")
    for relative in FILES:
        os.setxattr(SOURCE / relative, "system.posix_acl_access", read_acl(reader))
        print(f"Named reader {reader} granted read-only: {relative}", flush=True)


if __name__ == "__main__":
    main()
