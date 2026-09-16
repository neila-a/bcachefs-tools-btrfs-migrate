#!/usr/bin/env python3
"""Safe entry point for a real bcachefs migration invocation.

The runner deliberately refuses known test mounts, snapshots the executable, and
uses Btrfs ioctls from every visible mount namespace before invoking migrate.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Iterable


BTRFS_IOCTL_MAGIC = 0x94
IOC_READ = 2
IOC_READ_WRITE = 3
IOC_SIZE_SHIFT = 16
IOC_TYPE_SHIFT = 8
IOC_DIR_SHIFT = 30
BTRFS_FS_INFO_SIZE = 1024
BTRFS_DEV_INFO_SIZE = 4096
CLONE_NEWNS = 0x00020000


def _ioc(direction: int, request_type: int, number: int, size: int) -> int:
    return (
        (direction << IOC_DIR_SHIFT)
        | (request_type << IOC_TYPE_SHIFT)
        | number
        | (size << IOC_SIZE_SHIFT)
    )


BTRFS_IOC_DEV_INFO = _ioc(IOC_READ_WRITE, BTRFS_IOCTL_MAGIC, 30, BTRFS_DEV_INFO_SIZE)
BTRFS_IOC_FS_INFO = _ioc(IOC_READ, BTRFS_IOCTL_MAGIC, 31, BTRFS_FS_INFO_SIZE)


@dataclass(frozen=True)
class MountInfo:
    mount_point: str
    source: str


@dataclass(frozen=True)
class BtrfsMount:
    namespace: str
    mount_point: str
    fsid: str
    devices: tuple[str, ...]


class UnsafeTestMountError(RuntimeError):
    pass


class NamespaceDiscoveryError(RuntimeError):
    pass


def decode_mountinfo_path(value: str) -> str:
    return re.sub(
        r"\\([0-7]{3})",
        lambda match: chr(int(match.group(1), 8)),
        value,
    )


def parse_btrfs_mountinfo(mountinfo: str) -> list[MountInfo]:
    mounts = []
    for line in mountinfo.splitlines():
        separator = line.find(" - ")
        if separator < 0:
            continue
        left = line[:separator].split()
        right = line[separator + 3 :].split()
        if len(left) < 5 or len(right) < 2 or right[0] != "btrfs":
            continue
        mounts.append(
            MountInfo(
                mount_point=decode_mountinfo_path(left[4]),
                source=decode_mountinfo_path(right[1]),
            )
        )
    return mounts


def _fs_info(fd: int) -> tuple[int, str]:
    payload = bytearray(BTRFS_FS_INFO_SIZE)
    fcntl.ioctl(fd, BTRFS_IOC_FS_INFO, payload, True)
    max_id, _num_devices = struct.unpack_from("QQ", payload)
    fsid = str(uuid.UUID(bytes=bytes(payload[16:32])))
    return max_id, fsid


def _device_info(fd: int, devid: int) -> str:
    payload = bytearray(BTRFS_DEV_INFO_SIZE)
    struct.pack_into("Q", payload, 0, devid)
    fcntl.ioctl(fd, BTRFS_IOC_DEV_INFO, payload, True)
    return bytes(payload[3072:]).split(b"\0", 1)[0].decode(errors="surrogateescape")


def inspect_btrfs_mount(namespace: str, mount: MountInfo) -> BtrfsMount:
    fd = os.open(mount.mount_point, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        max_id, fsid = _fs_info(fd)
        devices = []
        for devid in range(1, max_id + 1):
            try:
                path = _device_info(fd, devid)
            except OSError as error:
                if error.errno in (errno.ENODEV, errno.ENOENT):
                    continue
                raise
            if path:
                devices.append(path)
        if not devices:
            raise NamespaceDiscoveryError(
                f"BTRFS_IOC_DEV_INFO returned no devices for {mount.mount_point}"
            )
        return BtrfsMount(namespace, mount.mount_point, fsid, tuple(devices))
    finally:
        os.close(fd)


def inspect_current_namespace(namespace: str) -> list[BtrfsMount]:
    mounts = parse_btrfs_mountinfo(Path("/proc/self/mountinfo").read_text())
    ensure_no_test_btrfs_mountinfo(namespace, mounts)
    return [inspect_btrfs_mount(namespace, mount) for mount in mounts]


def _set_mount_namespace(fd: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.setns(fd, CLONE_NEWNS) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def inspect_foreign_namespace(
    namespace_path: Path, mountinfo_path: Path, namespace: str
) -> list[BtrfsMount]:
    mounts = parse_btrfs_mountinfo(mountinfo_path.read_text())
    ensure_no_test_btrfs_mountinfo(namespace, mounts)
    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(read_fd)
        try:
            namespace_fd = os.open(namespace_path, os.O_RDONLY | os.O_CLOEXEC)
            try:
                _set_mount_namespace(namespace_fd)
            finally:
                os.close(namespace_fd)
            payload = {
                "mounts": [
                    asdict(inspect_btrfs_mount(namespace, mount)) for mount in mounts
                ]
            }
        except BaseException as error:
            payload = {"error": f"{type(error).__name__}: {error}"}
        encoded = memoryview(json.dumps(payload).encode())
        while encoded:
            written = os.write(write_fd, encoded)
            encoded = encoded[written:]
        os.close(write_fd)
        os._exit(0)

    os.close(write_fd)
    chunks = []
    while data := os.read(read_fd, 64 * 1024):
        chunks.append(data)
    os.close(read_fd)
    _pid, status = os.waitpid(child, 0)
    if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
        raise NamespaceDiscoveryError(f"namespace probe {namespace} exited unexpectedly")
    payload = json.loads(b"".join(chunks))
    if "error" in payload:
        raise NamespaceDiscoveryError(f"cannot inspect {namespace}: {payload['error']}")
    return [
        BtrfsMount(
            namespace=namespace,
            mount_point=mount["mount_point"],
            fsid=mount["fsid"],
            devices=tuple(mount["devices"]),
        )
        for mount in payload["mounts"]
    ]


def visible_mount_namespaces() -> dict[str, tuple[Path, Path]]:
    namespaces: dict[str, tuple[Path, Path]] = {}
    current_namespace = os.readlink("/proc/self/ns/mnt")
    namespaces[current_namespace] = (
        Path("/proc/self/ns/mnt"),
        Path("/proc/self/mountinfo"),
    )
    for process_path in Path("/proc").iterdir():
        if not process_path.name.isdecimal():
            continue
        namespace_path = process_path / "ns/mnt"
        try:
            namespace = os.readlink(namespace_path)
        except FileNotFoundError:
            continue
        except PermissionError as error:
            raise NamespaceDiscoveryError(
                f"cannot inspect mount namespace for PID {process_path.name}: {error}"
            ) from error
        namespaces.setdefault(
            namespace,
            (namespace_path, process_path / "mountinfo"),
        )
    return namespaces


def discover_btrfs_mounts() -> list[BtrfsMount]:
    current_namespace = os.readlink("/proc/self/ns/mnt")
    mounts = []
    for namespace, (namespace_path, mountinfo_path) in visible_mount_namespaces().items():
        if namespace == current_namespace:
            mounts.extend(inspect_current_namespace(namespace))
        else:
            mounts.extend(
                inspect_foreign_namespace(namespace_path, mountinfo_path, namespace)
            )
    return mounts


def is_test_btrfs_mount(mount_point: str, devices: Iterable[str]) -> bool:
    return (
        "/pytest-of-" in mount_point
        or "/pytest-" in mount_point
        or any(Path(device).name.startswith("loop") for device in devices)
    )


def ensure_no_test_btrfs_mountinfo(namespace: str, mounts: Iterable[MountInfo]) -> None:
    unsafe = [
        f"namespace={namespace} mount={mount.mount_point} devices={mount.source}"
        for mount in mounts
        if is_test_btrfs_mount(mount.mount_point, (mount.source,))
    ]
    if unsafe:
        raise UnsafeTestMountError(
            "refusing migration while pytest or loop-backed Btrfs mounts are inherited: "
            + "; ".join(unsafe)
        )


def ensure_no_test_btrfs_mounts(mounts: Iterable[BtrfsMount]) -> None:
    unsafe = [
        f"namespace={mount.namespace} mount={mount.mount_point} devices={','.join(mount.devices)}"
        for mount in mounts
        if is_test_btrfs_mount(mount.mount_point, mount.devices)
    ]
    if unsafe:
        raise UnsafeTestMountError(
            "refusing migration while pytest or loop-backed Btrfs mounts are inherited: "
            + "; ".join(unsafe)
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()


def bcachefs_version(binary: Path) -> str:
    output = command_output([os.fspath(binary), "version"])
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("bcachefs version produced no version line")
    return lines[-1]


def elf_build_id(binary: Path) -> str:
    readelf = shutil.which("readelf")
    if readelf is None:
        raise RuntimeError("readelf is required to record the migration binary build ID")
    result = subprocess.run(
        [readelf, "-n", os.fspath(binary)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        return "not-elf"
    match = re.search(r"Build ID:\s*([0-9A-Fa-f]+)", result.stdout)
    return match.group(1).lower() if match else "absent"


def copy_binary(source: Path, destination: Path) -> None:
    source_digest_before = sha256_file(source)
    source_fd = os.open(source, os.O_RDONLY | os.O_CLOEXEC)
    try:
        destination_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o500,
        )
        try:
            while block := os.read(source_fd, 1024 * 1024):
                offset = 0
                while offset < len(block):
                    offset += os.write(destination_fd, block[offset:])
            os.fchmod(destination_fd, 0o500)
        finally:
            os.close(destination_fd)
    finally:
        os.close(source_fd)
    source_digest_after = sha256_file(source)
    destination_digest = sha256_file(destination)
    if source_digest_before != source_digest_after or source_digest_before != destination_digest:
        raise RuntimeError("bcachefs build artifact changed while creating its run snapshot")


def git_head(repository: Path) -> str:
    return command_output(["git", "-C", os.fspath(repository), "rev-parse", "HEAD"])


def create_run_snapshot(source: Path, run_dir: Path) -> tuple[Path, dict[str, object]]:
    source = source.absolute()
    run_dir.mkdir(mode=0o700)
    binary_dir = run_dir / "binary"
    binary_dir.mkdir(mode=0o700)
    snapshot = binary_dir / "bcachefs"
    copy_binary(source, snapshot)
    binary_dir.chmod(0o500)
    manifest = {
        "created_unix_seconds": time.time(),
        "git_head": git_head(Path(__file__).parents[1]),
        "binary": {
            "source": os.fspath(source),
            "snapshot": os.fspath(snapshot),
            "sha256": sha256_file(snapshot),
            "version": bcachefs_version(snapshot),
            "build_id": elf_build_id(snapshot),
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return snapshot, manifest


def verify_snapshot(snapshot: Path, manifest: dict[str, object]) -> None:
    binary = manifest["binary"]
    assert isinstance(binary, dict)
    expected = binary["sha256"]
    assert isinstance(expected, str)
    if sha256_file(snapshot) != expected:
        raise RuntimeError("run snapshot no longer matches its recorded SHA256")
    if stat.S_IMODE(snapshot.stat().st_mode) != 0o500:
        raise RuntimeError("run snapshot permissions changed after verification")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Snapshot bcachefs then run migration under the watchdog."
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--watchdog", type=Path, default=Path(__file__).with_name("migrate-watchdog.py")
    )
    parser.add_argument("migration_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.migration_args:
        parser.error("migration arguments are required after --")
    if args.migration_args[0] == "--":
        args.migration_args.pop(0)
    if not args.migration_args:
        parser.error("migration arguments are required after --")
    return args


def main(args: argparse.Namespace) -> int:
    mounts = discover_btrfs_mounts()
    ensure_no_test_btrfs_mounts(mounts)
    snapshot, manifest = create_run_snapshot(args.binary, args.run_dir)
    verify_snapshot(snapshot, manifest)
    return subprocess.run(
        [
            sys.executable,
            os.fspath(args.watchdog),
            "--log-file",
            os.fspath(args.run_dir / "migrate-watchdog.log"),
            "--",
            os.fspath(snapshot),
            "migrate",
            *args.migration_args,
            "-f",
            os.fspath(args.source),
        ],
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
