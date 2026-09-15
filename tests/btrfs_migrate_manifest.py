#!/usr/bin/env python3
"""Create or verify a migration manifest without following symlinks."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import sys
import subprocess
from pathlib import Path
from typing import Any


CHUNK_SIZE = 1024 * 1024

POSIX_ACL_XATTR_VERSION = 0x0002
POSIX_ACL_XATTR_HEADER_SIZE = 4
POSIX_ACL_XATTR_ENTRY_SIZE = 8
ACL_UNDEFINED_ID = 0xFFFFFFFF

ACL_XATTR_NAMES = {
    b"system.posix_acl_access": "acl_access",
    b"system.posix_acl_default": "acl_default",
}
ACL_TAG_NAMES = {
    0x01: "user_obj",
    0x02: "user",
    0x04: "group_obj",
    0x08: "group",
    0x10: "mask",
    0x20: "other",
}
ACL_SHORT_TAGS = frozenset({0x01, 0x04, 0x10, 0x20})
ACL_NAMED_TAGS = frozenset({0x02, 0x08})
ACL_MANIFEST_KEYS = ("acl_access", "acl_default")
ACL_TAG_ORDER = {
    "user_obj": 0,
    "user": 1,
    "group_obj": 2,
    "group": 3,
    "mask": 4,
    "other": 5,
}


def _canonicalize_acl_entries(entries: list[list[Any]]) -> list[list[Any]]:
    return sorted(
        entries,
        key=lambda entry: (
            ACL_TAG_ORDER[entry[0]],
            -1 if entry[2] is None else entry[2],
        ),
    )


def _parse_linux_posix_acl_xattr(value: bytes) -> list[list[Any]]:
    if len(value) < POSIX_ACL_XATTR_HEADER_SIZE:
        raise ValueError(
            "Linux POSIX ACL xattr is shorter than its "
            f"{POSIX_ACL_XATTR_HEADER_SIZE}-byte header ({len(value)} bytes)"
        )

    version = int.from_bytes(value[:4], "little")
    if version != POSIX_ACL_XATTR_VERSION:
        raise ValueError(f"unsupported Linux POSIX ACL xattr version {version}")

    payload_len = len(value) - POSIX_ACL_XATTR_HEADER_SIZE
    if payload_len % POSIX_ACL_XATTR_ENTRY_SIZE:
        raise ValueError(
            "Linux POSIX ACL xattr payload is "
            f"{payload_len} bytes, not divisible by {POSIX_ACL_XATTR_ENTRY_SIZE}"
        )

    entries: list[list[Any]] = []
    for entry_index in range(payload_len // POSIX_ACL_XATTR_ENTRY_SIZE):
        offset = POSIX_ACL_XATTR_HEADER_SIZE + entry_index * POSIX_ACL_XATTR_ENTRY_SIZE
        entry = value[offset : offset + POSIX_ACL_XATTR_ENTRY_SIZE]
        if len(entry) != POSIX_ACL_XATTR_ENTRY_SIZE:
            raise ValueError(f"POSIX ACL entry {entry_index} is truncated")

        tag = int.from_bytes(entry[:2], "little")
        perm = int.from_bytes(entry[2:4], "little")
        entry_id = int.from_bytes(entry[4:8], "little")
        if perm & ~0x7:
            raise ValueError(
                f"Linux POSIX ACL entry {entry_index} has invalid permissions "
                f"0x{perm:x}"
            )

        tag_name = ACL_TAG_NAMES.get(tag)
        if tag_name is None:
            raise ValueError(
                f"Linux POSIX ACL entry {entry_index} has unknown tag 0x{tag:x}"
            )
        if tag in ACL_SHORT_TAGS:
            if entry_id != ACL_UNDEFINED_ID:
                raise ValueError(
                    f"Linux POSIX ACL entry {entry_index} has concrete id "
                    f"{entry_id} for short tag 0x{tag:x}"
                )
            canonical_id = None
        elif tag in ACL_NAMED_TAGS:
            if entry_id == ACL_UNDEFINED_ID:
                raise ValueError(
                    f"Linux POSIX ACL entry {entry_index} has undefined id "
                    f"for named tag 0x{tag:x}"
                )
            canonical_id = entry_id
        else:
            raise ValueError(
                f"Linux POSIX ACL entry {entry_index} has unsupported tag 0x{tag:x}"
            )

        entries.append([tag_name, perm, canonical_id])

    return _canonicalize_acl_entries(entries)


def _xattrs(
    path: bytes, *, include_acls: bool
) -> tuple[dict[str, str], dict[str, list[list[Any]]]]:
    names = sorted(os.listxattr(path, follow_symlinks=False))
    xattrs: dict[str, str] = {}
    acls: dict[str, list[list[Any]]] = {}
    for name in names:
        name_bytes = os.fsencode(name)
        acl_key = ACL_XATTR_NAMES.get(name_bytes)
        if acl_key is not None:
            if include_acls:
                value = os.getxattr(path, name, follow_symlinks=False)
                acls[acl_key] = _parse_linux_posix_acl_xattr(value)
            continue
        value = os.getxattr(path, name, follow_symlinks=False)
        xattrs[name_bytes.hex()] = value.hex()
    return xattrs, acls




def _sha256(path: bytes) -> str:
    digest = hashlib.sha256()
    with open(path, "rb", buffering=0) as stream:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _entry(root: bytes, path: bytes, *, include_acls: bool) -> dict[str, Any]:
    info = os.lstat(path)
    relative = os.path.relpath(path, root)
    xattrs, acls = _xattrs(path, include_acls=include_acls)
    item: dict[str, Any] = {
        "path": base64.b64encode(relative).decode("ascii"),
        "mode": stat.S_IMODE(info.st_mode),
        "uid": info.st_uid,
        "gid": info.st_gid,
        "mtime_ns": info.st_mtime_ns,
        "xattrs": xattrs,
    }
    item.update(acls)
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


def _manifest(root: Path, *, include_acls: bool) -> dict[str, Any]:
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
            item = _entry(root_bytes, path, include_acls=include_acls)
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

def _parse_getfacl_acls(output: bytes) -> dict[str, list[list[Any]]]:
    acls = {key: [] for key in ACL_MANIFEST_KEYS}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(b"#"):
            continue

        acl_key = "acl_access"
        if line.startswith(b"default:"):
            acl_key = "acl_default"
            line = line[len(b"default:") :]

        line = line.split(b"#", 1)[0].rstrip()
        fields = line.split(b":")
        if len(fields) != 3:
            raise ValueError(f"unexpected getfacl entry: {raw_line!r}")
        tag, qualifier, permissions = fields
        permissions = permissions.split(b"#", 1)[0]
        if len(permissions) != 3 or any(
            char not in b"rwx-" for char in permissions
        ):
            raise ValueError(f"invalid getfacl permissions: {raw_line!r}")
        perm = sum(
            bit
            for char, bit in zip(permissions, (4, 2, 1))
            if char != ord("-")
        )

        if qualifier:
            if not qualifier.isdigit():
                raise ValueError(f"non-numeric getfacl qualifier: {raw_line!r}")
            entry_id: int | None = int(qualifier)
        else:
            entry_id = None

        if tag == b"user":
            entry_name = "user" if qualifier else "user_obj"
        elif tag == b"group":
            entry_name = "group" if qualifier else "group_obj"
        elif tag == b"mask":
            if qualifier:
                raise ValueError(f"mask getfacl entry has a qualifier: {raw_line!r}")
            entry_name = "mask"
            entry_id = None
        elif tag == b"other":
            if qualifier:
                raise ValueError(f"other getfacl entry has a qualifier: {raw_line!r}")
            entry_name = "other"
            entry_id = None
        else:
            raise ValueError(f"unknown getfacl tag: {raw_line!r}")

        acls[acl_key].append([entry_name, perm, entry_id])
    return {
        key: _canonicalize_acl_entries(entries) for key, entries in acls.items()
    }


def _getfacl_acls(path: bytes) -> dict[str, list[list[Any]]]:
    result = subprocess.run(
        [b"getfacl", b"-cpn", path],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        if not detail:
            detail = f"exit status {result.returncode}"
        raise OSError(f"getfacl failed for {path!r}: {detail}")
    return _parse_getfacl_acls(result.stdout)


def _without_acls(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in entry.items() if key not in ACL_MANIFEST_KEYS
    }


def _verify_acl_manifest(root: Path, expected: dict[str, Any]) -> bool:
    root_bytes = os.fsencode(root)
    matches = True
    for entry in expected.get("entries", []):
        acl_keys = [key for key in ACL_MANIFEST_KEYS if key in entry]
        if not acl_keys:
            continue

        relative = base64.b64decode(entry["path"], validate=True)
        path = os.path.join(root_bytes, relative)
        actual = _getfacl_acls(path)
        for acl_key in acl_keys:
            if actual[acl_key] == entry[acl_key]:
                continue
            print(
                f"MANIFEST_ACL_MISMATCH {entry['path']} {acl_key}",
                file=sys.stderr,
            )
            print(f"expected={entry[acl_key]!r}", file=sys.stderr)
            print(f"actual={actual[acl_key]!r}", file=sys.stderr)
            matches = False
    return matches


def main(argv: list[str]) -> int:
    if len(argv) != 4 or argv[1] not in {"create", "verify"}:
        print(f"usage: {argv[0]} create|verify ROOT MANIFEST", file=sys.stderr)
        return 2

    operation, root_text, manifest_text = argv[1:]
    root = Path(root_text)
    manifest_path = Path(manifest_text)
    actual = _manifest(root, include_acls=operation == "create")
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
    expected_without_acls = {
        **expected,
        "entries": [
            _without_acls(entry) for entry in expected.get("entries", [])
        ],
    }
    if expected_without_acls != actual:
        print("MANIFEST_MISMATCH", file=sys.stderr)
        expected_entries = {
            entry["path"]: entry for entry in expected_without_acls.get("entries", [])
        }
        actual_entries = {entry["path"]: entry for entry in actual["entries"]}
        for path in sorted(set(expected_entries) | set(actual_entries), key=os.fsencode):
            if expected_entries.get(path) != actual_entries.get(path):
                print(path, file=sys.stderr)
        return 1

    if not _verify_acl_manifest(root, expected):
        return 1

    print(f"MANIFEST_OK {root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"MANIFEST_ERROR {error}", file=sys.stderr)
        raise SystemExit(1)
