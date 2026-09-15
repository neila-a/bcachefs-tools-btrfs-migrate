#!/usr/bin/env python3
"""Privileged synthetic Btrfs-to-bcachefs migration fixtures.

Every fixture uses a disposable sparse image under pytest's temporary
workspace. It never selects or names a host block device.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pytest


ROOT = Path(__file__).resolve().parents[1]
BCACHEFS = Path(os.environ.get("BCACHEFS_BINARY", ROOT / "target/release/bcachefs"))
GIB = 1024**3
IMAGE_BYTES = int(os.environ.get("BCACHEFS_MIGRATE_IMAGE_BYTES", 12 * GIB))
MIGRATE_TIMEOUT_SECONDS = 5 * 60
FORBIDDEN_ALLOCATOR_OUTPUT = (
    "no_buckets_found",
    "Allocator stuck",
    "btree_cache_cannibalize_lock_blocked",
    "trans path overflow",
)
RESERVATION_FIELDS = (
    "reservation_raw_extent_count",
    "reservation_raw_bytes",
    "reservation_bucket_size_bytes",
    "reservation_usable_full_bucket_count",
    "reservation_usable_full_bucket_bytes",
    "reservation_bucket_alignment_loss_bytes",
)

TOP_LEVEL_RW_MOUNT_OPTIONS = "rw,noatime,subvolid=5"
TOP_LEVEL_RO_MOUNT_OPTIONS = "ro,noatime,subvolid=5"



def _has_cap_sys_admin() -> bool:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("CapEff:"):
            return bool(int(line.split()[1], 16) & (1 << 21))
    return False


def _require_host_prerequisites() -> None:
    if not _has_cap_sys_admin():
        pytest.skip("requires CAP_SYS_ADMIN for a disposable Btrfs loop mount")
    required = (
        "mkfs.btrfs",
        "btrfs",
        "blockdev",
        "losetup",
        "mount",
        "umount",
        "sha256sum",
    )
    missing = [command for command in required if shutil.which(command) is None]
    if missing:
        pytest.skip(f"missing Btrfs fixture commands: {', '.join(missing)}")
    if not BCACHEFS.is_file() or not os.access(BCACHEFS, os.X_OK):
        pytest.skip(f"bcachefs binary is not executable: {BCACHEFS}")


def _run(
    command: Iterable[os.PathLike[str] | str],
    *,
    check: bool = True,
    timeout: int = MIGRATE_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    command = [os.fspath(arg) for arg in command]
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        pytest.fail(
            f"command failed ({result.returncode}): {' '.join(command)}\n{result.stdout}"
        )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as image:
        for block in iter(lambda: image.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reservation_metrics(output: str) -> dict[str, int]:
    metrics = {
        name: int(match.group(1))
        for name in RESERVATION_FIELDS
        if (match := re.search(rf"(?:^|\s){name}=(\d+)(?:\s|$)", output))
    }
    missing = sorted(set(RESERVATION_FIELDS) - metrics.keys())
    assert not missing, f"missing reservation metrics {missing}:\n{output}"
    return metrics


def _assert_bounded_allocator_result(result: subprocess.CompletedProcess[str]) -> None:
    for marker in FORBIDDEN_ALLOCATOR_OUTPUT:
        assert marker not in result.stdout, result.stdout
    if result.returncode:
        assert re.search(r"(?im)^error(?:\[[^]]+\])?:", result.stdout), result.stdout



def _mount_options(mountpoint: Path) -> set[str]:
    target = os.path.realpath(mountpoint)
    for entry in Path("/proc/self/mountinfo").read_text().splitlines():
        fields = entry.split()
        if len(fields) < 6:
            continue
        mounted_at = re.sub(
            r"\\([0-7]{3})",
            lambda match: chr(int(match.group(1), 8)),
            fields[4],
        )
        if mounted_at == target:
            separator = fields.index("-")
            return set(fields[5].split(",")) | set(fields[separator + 3].split(","))
    raise AssertionError(f"mountpoint is not mounted: {mountpoint}")



@dataclass
class BtrfsImage:
    image: Path
    mountpoint: Path
    mounted: bool = False
    loop_device: Path | None = None

    @classmethod
    def create(cls, workspace: Path) -> "BtrfsImage":
        image = workspace / "source.btrfs"
        mountpoint = workspace / "mnt"
        mountpoint.mkdir()
        with image.open("wb") as file:
            file.truncate(IMAGE_BYTES)
        _run(["mkfs.btrfs", "-f", "-d", "single", "-m", "dup", image])
        fixture = cls(image, mountpoint)
        try:
            fixture.mount_readwrite_top_level()
            fixture.assert_single_device_profiles()
        except BaseException:
            fixture.cleanup()
            raise
        return fixture

    def _attach_loop(self) -> None:
        if self.loop_device is not None:
            return
        result = _run(
            ["losetup", "--find", "--show", "--nooverlap", self.image], check=False
        )
        if result.returncode:
            if re.search(
                r"permission denied|operation not permitted|权限不够|不允许",
                result.stdout,
                re.IGNORECASE,
            ):
                pytest.skip(f"Btrfs loop setup is unavailable: {result.stdout.strip()}")
            pytest.fail(
                f"losetup failed ({result.returncode}): {self.image}\n{result.stdout}"
            )
        self.loop_device = Path(result.stdout.strip())
        _run(["blockdev", "--setrw", self.loop_device])

    def _assert_loop_writable(self) -> None:
        assert self.loop_device is not None
        readonly = _run(
            ["losetup", "--noheadings", "--output", "RO", self.loop_device]
        ).stdout.strip()
        assert readonly == "0", f"loop device is read-only: {self.loop_device}"

    def _try_unmount(self) -> str | None:
        if not self.mounted:
            return None
        result = _run(["umount", self.mountpoint], check=False)
        if result.returncode:
            return f"umount failed ({result.returncode}): {result.stdout}"
        self.mounted = False
        return None

    def _try_detach_loop(self) -> str | None:
        if self.loop_device is None:
            return None
        result = _run(["losetup", "--detach", self.loop_device], check=False)
        if result.returncode:
            return (
                f"losetup --detach failed ({result.returncode}): {self.loop_device}\n"
                f"{result.stdout}"
            )
        self.loop_device = None
        return None

    def mount(self, options: str = TOP_LEVEL_RW_MOUNT_OPTIONS) -> None:
        assert not self.mounted
        option_set = set(options.split(","))
        assert len(option_set & {"ro", "rw"}) == 1, options
        self._attach_loop()
        assert self.loop_device is not None
        self._assert_loop_writable()
        loop_device = self.loop_device
        mount_mode = "--read-only" if "ro" in option_set else "--read-write"
        result = _run(
            [
                "mount",
                mount_mode,
                "-t",
                "btrfs",
                "-o",
                options,
                loop_device,
                self.mountpoint,
            ],
            check=False,
        )
        if result.returncode:
            cleanup_error = self._try_detach_loop()
            if cleanup_error:
                pytest.fail(cleanup_error)
            if re.search(
                r"permission denied|operation not permitted|权限不够|不允许",
                result.stdout,
                re.IGNORECASE,
            ):
                pytest.skip(f"Btrfs loop mount is unavailable: {result.stdout.strip()}")
            pytest.fail(
                f"mount failed ({result.returncode}): {loop_device} -> "
                f"{self.mountpoint}\n{result.stdout}"
            )
        self.mounted = True
        try:
            self._assert_top_level_mount_options("ro" if "ro" in option_set else "rw")
        except BaseException:
            cleanup_errors = [
                error
                for error in (self._try_unmount(), self._try_detach_loop())
                if error
            ]
            if cleanup_errors:
                pytest.fail("\n".join(cleanup_errors))
            raise

    def mount_readwrite_top_level(self) -> None:
        self.mount(TOP_LEVEL_RW_MOUNT_OPTIONS)

    def switch_to_readwrite_top_level(self) -> None:
        self.unmount()
        self.mount_readwrite_top_level()

    def switch_to_readonly_top_level(self) -> None:
        self.unmount()
        self.mount(TOP_LEVEL_RO_MOUNT_OPTIONS)

    def unmount(self) -> None:
        if error := self._try_unmount():
            pytest.fail(error)

    def _assert_top_level_mount_options(self, mode: str) -> None:
        options = _mount_options(self.mountpoint)
        required = {mode, "noatime", "subvolid=5"}
        missing = sorted(required - options)
        assert not missing, f"missing mount options {missing}: {sorted(options)}"
        assert ("ro" if mode == "rw" else "rw") not in options, sorted(options)

    def assert_writable_top_level(self) -> None:
        self._assert_top_level_mount_options("rw")

    def assert_readonly_top_level(self) -> None:
        self._assert_top_level_mount_options("ro")


    def sync(self) -> None:
        self.assert_writable_top_level()
        _run(["btrfs", "filesystem", "sync", self.mountpoint])

    def assert_single_device_profiles(self) -> None:
        filesystem_df = _run(["btrfs", "filesystem", "df", self.mountpoint]).stdout
        assert re.search(r"(?im)^Data, single:", filesystem_df), filesystem_df
        assert re.search(r"(?im)^Metadata, DUP:", filesystem_df), filesystem_df
        assert re.search(r"(?im)^System, DUP:", filesystem_df), filesystem_df

    def check_readonly(self) -> None:
        self.unmount()
        _run(["btrfs", "check", "--readonly", self.image], timeout=MIGRATE_TIMEOUT_SECONDS)

    def cleanup(self) -> None:
        errors = [
            error
            for error in (self._try_unmount(), self._try_detach_loop())
            if error
        ]
        try:
            self.image.unlink(missing_ok=True)
            self.mountpoint.rmdir()
        except OSError as error:
            errors.append(f"fixture path cleanup failed: {error}")
        if errors:
            pytest.fail("\n".join(errors))


@pytest.fixture
def btrfs_image(tmp_path: Path) -> Iterable[BtrfsImage]:
    _require_host_prerequisites()
    fixture = BtrfsImage.create(tmp_path)
    try:
        yield fixture
    finally:
        fixture.cleanup()


def test_default_mount_is_writable_top_level() -> None:
    assert BtrfsImage.mount.__defaults__ == ("rw,noatime,subvolid=5",)

def _write_tree(
    image: BtrfsImage, directories: int, files_per_directory: int
) -> None:
    image.assert_writable_top_level()
    for directory_index in range(directories):
        directory = image.mountpoint / f"dir-{directory_index:04d}"
        directory.mkdir()
        for file_index in range(files_per_directory):
            (directory / f"file-{file_index:04d}").write_bytes(
                f"{directory_index}:{file_index}\n".encode() * 64
            )


def _fragment_available_space(image: BtrfsImage) -> None:
    image.assert_writable_top_level()
    fragments = []
    for index in range(48):
        path = image.mountpoint / f"fragment-{index:03d}"
        with path.open("wb") as file:
            os.posix_fallocate(file.fileno(), 0, 4 * 1024 * 1024)
            file.flush()
            os.fsync(file.fileno())
        fragments.append(path)
    for path in fragments[::2]:
        path.unlink()
    image.sync()


def _create_child_subvolume(image: BtrfsImage) -> str:
    image.assert_writable_top_level()
    child_path = image.mountpoint / "child"
    _run(["btrfs", "subvolume", "create", child_path])
    return _run(["btrfs", "subvolume", "show", child_path]).stdout


def _migrate(image: BtrfsImage, *args: str) -> subprocess.CompletedProcess[str]:
    try:
        return _run([BCACHEFS, "migrate", *args, image.mountpoint], check=False)
    except subprocess.TimeoutExpired as error:
        pytest.fail(
            f"bcachefs migrate exceeded {MIGRATE_TIMEOUT_SECONDS}s: {error.cmd}"
        )


def test_a_supported_btrfs_no_change_is_eligible(btrfs_image: BtrfsImage) -> None:
    _write_tree(btrfs_image, directories=4, files_per_directory=8)
    btrfs_image.sync()
    btrfs_image.switch_to_readonly_top_level()
    btrfs_image.assert_readonly_top_level()
    image_before = _sha256(btrfs_image.image)

    result = _migrate(btrfs_image, "--no-change", "-f")

    assert result.returncode == 0, result.stdout
    assert "MIGRATE_RESULT eligible=yes" in result.stdout, result.stdout
    assert not (btrfs_image.mountpoint / "bcachefs").exists()
    assert _sha256(btrfs_image.image) == image_before


def test_b_supported_btrfs_migrates_with_full_reservation_metrics(
    btrfs_image: BtrfsImage,
) -> None:
    _write_tree(btrfs_image, directories=8, files_per_directory=16)
    btrfs_image.sync()
    btrfs_image.assert_writable_top_level()

    result = _migrate(btrfs_image, "-f")

    assert result.returncode == 0, result.stdout
    assert "MIGRATE_RESULT eligible=yes" in result.stdout, result.stdout
    _assert_bounded_allocator_result(result)
    metrics = _reservation_metrics(result.stdout)
    assert metrics["reservation_usable_full_bucket_bytes"] > 0
    assert (btrfs_image.mountpoint / "bcachefs").is_file()
    btrfs_image.check_readonly()


def test_c_child_subvolume_no_change_is_rejected_without_reservation(
    btrfs_image: BtrfsImage,
) -> None:
    child = _create_child_subvolume(btrfs_image)
    assert re.search(r"(?m)^\s*Subvolume ID:\s+256$", child), child
    btrfs_image.sync()
    btrfs_image.switch_to_readonly_top_level()
    btrfs_image.assert_readonly_top_level()
    image_before = _sha256(btrfs_image.image)

    result = _migrate(btrfs_image, "--no-change", "-f")

    assert result.returncode != 0, result.stdout
    assert "MIGRATE_RESULT eligible=no" in result.stdout, result.stdout
    assert "child subvolume" in result.stdout.lower(), result.stdout
    assert "Root Tree returned a record outside the search key" not in result.stdout
    assert not (btrfs_image.mountpoint / "bcachefs").exists()
    assert _sha256(btrfs_image.image) == image_before


def test_stress_migrate_is_bounded_and_avoids_allocator_path_leaks(
    btrfs_image: BtrfsImage,
) -> None:
    _fragment_available_space(btrfs_image)
    _write_tree(btrfs_image, directories=128, files_per_directory=48)
    btrfs_image.sync()
    btrfs_image.assert_writable_top_level()

    result = _migrate(btrfs_image, "-f")

    _assert_bounded_allocator_result(result)
    if result.returncode == 0:
        metrics = _reservation_metrics(result.stdout)
        assert metrics["reservation_usable_full_bucket_bytes"] > 0, result.stdout
        assert "MIGRATE_RESULT eligible=yes" in result.stdout, result.stdout
