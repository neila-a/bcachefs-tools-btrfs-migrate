#!/usr/bin/env python3
"""Create or verify a migration manifest without following symlinks."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any


CHUNK_SIZE = 1024 * 1024


def _xattrs(path: bytes) -> dict[str, str]:
    names = sorted(os.listxattr(path, follow_symlinks=False))
    return {
        os.fsencode(name).hex(): os.getxattr(path, name, follow_symlinks=False).hex()
        for name in names
    }


def _sha256(path: bytes) -> str:
    digest = hashlib.sha256()
    with open(path, "rb", buffering=0) as stream:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _entry(root: bytes, path: bytes) -> dict[str, Any]:
    info = os.lstat(path)
    relative = os.path.relpath(path, root)
    item: dict[str, Any] = {
        "path": base64.b64encode(relative).decode("ascii"),
        "mode": stat.S_IMODE(info.st_mode),
        "uid": info.st_uid,
        "gid": info.st_gid,
        "mtime_ns": info.st_mtime_ns,
        "xattrs": _xattrs(path),
    }
    mode = info.st_mode
    if stat.S_ISDIR(mode):
        item["type"] = "directory"
    elif stat.S_ISREG(mode):
        item["type"] = "file"
        item["size"] = info.st_size
        item["sha256"] = _sha256(path)
        if info.st_nlink > 1:
            item["_hardlink_key"] = [info.st_dev, info.st_ino]
    elif stat.S_ISLNK(mode):
        item["type"] = "symlink"
        item["target"] = os.fsencode(os.readlink(path)).hex()
    elif stat.S_ISFIFO(mode):
        item["type"] = "fifo"
    elif stat.S_ISSOCK(mode):
        item["type"] = "socket"
    elif stat.S_ISBLK(mode):
        item["type"] = "block"
        item["rdev"] = info.st_rdev
    elif stat.S_ISCHR(mode):
        item["type"] = "character"
        item["rdev"] = info.st_rdev
    else:
        item["type"] = "other"
    return item


def _manifest(root: Path) -> dict[str, Any]:
    root_bytes = os.fsencode(root)
    root_info = os.lstat(root_bytes)
    if not stat.S_ISDIR(root_info.st_mode):
        raise ValueError(f"manifest root is not a directory: {root}")

    entries: list[dict[str, Any]] = []
    pending = [root_bytes]
    while pending:
        directory = pending.pop()
        children = sorted(os.scandir(directory), key=lambda entry: os.fsencode(entry.name))
        for child in children:
            if directory == root_bytes and child.name in {
                b"bcachefs",
                b"old_migrated_filesystem",
            }:
                continue
            path = os.fsencode(child.path)
            item = _entry(root_bytes, path)
            entries.append(item)
            if item["type"] == "directory":
                pending.append(path)

    entries.sort(key=lambda item: os.fsencode(item["path"]))
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for item in entries:
        key = item.pop("_hardlink_key", None)
        if key is not None:
            groups.setdefault((key[0], key[1]), []).append(item)
    ordered_groups = sorted(
        (members for members in groups.values() if len(members) > 1),
        key=lambda members: min(os.fsencode(item["path"]) for item in members),
    )
    for group_id, members in enumerate(ordered_groups, 1):
        for item in members:
            item["hardlink"] = group_id
    return {"version": 1, "path_encoding": "base64", "entries": entries}


def main(argv: list[str]) -> int:
    if len(argv) != 4 or argv[1] not in {"create", "verify"}:
        print(f"usage: {argv[0]} create|verify ROOT MANIFEST", file=sys.stderr)
        return 2

    operation, root_text, manifest_text = argv[1:]
    root = Path(root_text)
    manifest_path = Path(manifest_text)
    actual = _manifest(root)
    if operation == "create":
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest_path.with_name(manifest_path.name + ".tmp")
        temporary.write_text(
            json.dumps(actual, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, manifest_path)
        print(f"MANIFEST_CREATED {manifest_path}")
        return 0

    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    if expected != actual:
        print("MANIFEST_MISMATCH", file=sys.stderr)
        expected_entries = {entry["path"]: entry for entry in expected.get("entries", [])}
        actual_entries = {entry["path"]: entry for entry in actual["entries"]}
        for path in sorted(set(expected_entries) | set(actual_entries), key=os.fsencode):
            if expected_entries.get(path) != actual_entries.get(path):
                print(path, file=sys.stderr)
        return 1

    print(f"MANIFEST_OK {root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"MANIFEST_ERROR {error}", file=sys.stderr)
        raise SystemExit(1)
