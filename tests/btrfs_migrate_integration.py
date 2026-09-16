#!/usr/bin/env python3
"""Privileged synthetic Btrfs-to-bcachefs migration fixtures.

Every fixture uses a disposable sparse image under pytest's temporary
workspace. It never selects or names a host block device.
"""

from __future__ import annotations

import hashlib
import os
import re
import signal
import stat
import select
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pytest


ROOT = Path(__file__).resolve().parents[1]
BCACHEFS = Path(os.environ.get("BCACHEFS_BINARY", ROOT / "target/release/bcachefs"))
GIB = 1024**3
IMAGE_BYTES = int(os.environ.get("BCACHEFS_MIGRATE_IMAGE_BYTES", 12 * GIB))
MIGRATE_TIMEOUT_SECONDS = 5 * 60
GOING_RO_ABORT_TIMEOUT_SECONDS = 5
CANDIDATE_MOUNT_TIMEOUT_SECONDS = 30

FORBIDDEN_ALLOCATOR_OUTPUT = (
    "no_buckets_found",
    "Allocator stuck",
    "btree_cache_cannibalize_lock_blocked",
    "trans path overflow",
    "path 0 refcount overflow",
    "__btree_path_get",
)
RESERVATION_FIELDS = (
    "reservation_raw_extent_count",
    "reservation_raw_bytes",
    "reservation_bucket_size_bytes",
    "reservation_usable_full_bucket_count",
    "reservation_usable_full_bucket_bytes",
    "reservation_bucket_alignment_loss_bytes",
    "reservation_default_superblock_protected_bytes",
    "reservation_device_tail_bytes",
)

RESERVATION_NOUSE_FIELDS = (
    "reservation_total_bucket_count",
    "reservation_nouse_excluded_bucket_count",
    "reservation_allocatable_bucket_count",
    "reservation_allocatable_bytes",
)
ALTERNATE_SUPERBLOCK_RE = re.compile(r"mount -t bcachefs -o ro,sb=(\d+)")



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
        "cp",
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


def _reservation_nouse_metrics(output: str) -> dict[str, int]:
    match = re.search(r"(?m)^MIGRATE_RESERVATION_NOUSE (?P<fields>.+)$", output)
    assert match, f"missing reservation nouse metrics:\n{output}"
    fields = dict(field.split("=", 1) for field in match["fields"].split())
    return {name: int(fields[name]) for name in RESERVATION_NOUSE_FIELDS}

def _allocator_scan_metrics(output: str) -> dict[str, int]:
    match = re.search(
        r"(?m)^MIGRATE_ALLOCATOR_SCAN max_paths=(\d+) "
        r"zero_ref_preserved_freespace_paths=(\d+) scans=(\d+) "
        r"injected_restarts=(\d+)$",
        output,
    )
    assert match, f"missing allocator scan metrics:\n{output}"
    return {
        "max_paths": int(match.group(1)),
        "zero_ref_preserved_freespace_paths": int(match.group(2)),
        "scans": int(match.group(3)),
        "injected_restarts": int(match.group(4)),
    }

PROGRESS_FIELDS = (
    "entries_done",
    "entries_total",
    "non_directories_done",
    "directories_done",
    "linked_bytes",
    "copied_bytes",
    "elapsed_seconds",
    "reservation_allocatable_bytes",
    "reservation_copied_bytes",
    "reservation_predicted_copy_bytes",
    "reservation_allocator_headroom_bytes",
    "reservation_required_remaining_bytes",
    "reservation_remaining_bytes",
)

COPY_REASON_FIELDS = (
    "copied_shared_bytes",
    "copied_inline_bytes",
    "copied_encoded_bytes",
    "copied_encrypted_bytes",
    "copied_not_aligned_bytes",
    "copied_tail_bytes",
    "copied_merged_bytes",
    "copied_future_flags_bytes",
    "copied_reservation_overlap_bytes",
    "copied_raw_unaligned_bytes",
    "copied_raw_mapping_unavailable_bytes",
    "copied_unstable_bytes",
    "copied_symlink_bytes",
    "copied_other_bytes",
)


def _copy_reasons(output: str, mode: str) -> dict[str, int]:
    match = re.search(
        rf"(?m)^MIGRATE_COPY_REASONS mode={re.escape(mode)} (?P<fields>.+)$",
        output,
    )
    assert match, f"missing {mode} copy reason accounting:\n{output}"
    fields = dict(field.split("=", 1) for field in match["fields"].split())
    parsed = {name: int(fields[name]) for name in COPY_REASON_FIELDS}
    assert int(fields["copied_bytes"]) == sum(parsed.values()), output
    return parsed


def _copy_policy(output: str, mode: str) -> dict[str, int]:
    match = re.search(
        rf"(?m)^MIGRATE_COPY_POLICY mode={re.escape(mode)} (?P<fields>.+)$",
        output,
    )
    assert match, f"missing {mode} copy policy:\n{output}"
    fields = dict(field.split("=", 1) for field in match["fields"].split())
    return {
        "candidate_block_bytes": int(fields["candidate_block_bytes"]),
        "reserved_superblock_end": int(fields["reserved_superblock_end"]),
    }

PLAN_FIELDS = (
    "linked_bytes",
    "copied_bytes",
    "unwritten_hole_bytes",
    "unstable_extent_bytes",
    "mapped_extent_count",
    "mapped_segment_count",
    "chunk_split_count",
)


def _migration_plan(output: str, mode: str) -> dict[str, int]:
    match = re.search(
        rf"(?m)^MIGRATE_PLAN mode={re.escape(mode)} (?P<fields>.+)$",
        output,
    )
    assert match, f"missing {mode} migration plan:\n{output}"
    fields = dict(field.split("=", 1) for field in match["fields"].split())
    return {name: int(fields[name]) for name in PLAN_FIELDS}




def _migration_progress(output: str) -> list[dict[str, int]]:
    progress = [
        dict(field.split("=", 1) for field in match["fields"].split())
        for match in re.finditer(r"(?m)^MIGRATE_PROGRESS (?P<fields>.+)$", output)
    ]
    assert progress, f"missing migration progress:\n{output}"
    parsed = [{name: int(record[name]) for name in PROGRESS_FIELDS} for record in progress]
    assert all(record["entries_total"] > 0 for record in parsed), output
    assert len({record["entries_total"] for record in parsed}) == 1, output
    for field in (
        "entries_done",
        "non_directories_done",
        "directories_done",
        "elapsed_seconds",
        "linked_bytes",
        "copied_bytes",
    ):
        assert all(
            current[field] >= previous[field]
            for previous, current in zip(parsed, parsed[1:])
        ), output
    assert all(
        current["linked_bytes"] + current["copied_bytes"]
        >= previous["linked_bytes"] + previous["copied_bytes"]
        for previous, current in zip(parsed, parsed[1:])
    ), output
    assert all(
        record["reservation_allocatable_bytes"] > 0
        and record["reservation_copied_bytes"] == record["copied_bytes"]
        and record["reservation_remaining_bytes"]
        == record["reservation_allocatable_bytes"] - record["copied_bytes"]
        and record["reservation_predicted_copy_bytes"]
        + record["reservation_allocator_headroom_bytes"]
        <= record["reservation_allocatable_bytes"]
        for record in parsed
    ), output
    return parsed


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
    candidate_mountpoint: Path | None = None
    candidate_process: subprocess.Popen[str] | None = None

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
        assert self.candidate_mountpoint is None
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
        assert self.loop_device is not None
        _run(
            ["btrfs", "check", "--readonly", self.loop_device],
            timeout=MIGRATE_TIMEOUT_SECONDS,
        )

    def mount_bcachefs_candidate_readonly(self, alternate_sb: int) -> Path:
        assert self.loop_device is not None
        assert self.candidate_mountpoint is None
        assert self.candidate_process is None
        self.unmount()
        mountpoint = self.image.parent / "candidate"
        mountpoint.mkdir()
        process = subprocess.Popen(
            [
                BCACHEFS,
                "fusemount",
                "-f",
                "-o",
                f"ro,sb={alternate_sb}",
                self.loop_device,
                mountpoint,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self.candidate_process = process
        for _ in range(CANDIDATE_MOUNT_TIMEOUT_SECONDS * 10):
            if process.poll() is not None:
                self.candidate_process = None
                mountpoint.rmdir()
                pytest.fail(
                    f"bcachefs candidate mount exited before readiness: {process.returncode}"
                )
            try:
                options = _mount_options(mountpoint)
            except AssertionError:
                time.sleep(0.1)
                continue
            assert "ro" in options, options
            self.candidate_mountpoint = mountpoint
            return mountpoint
        process.terminate()
        process.wait(timeout=CANDIDATE_MOUNT_TIMEOUT_SECONDS)
        self.candidate_process = None
        mountpoint.rmdir()
        pytest.fail("bcachefs candidate mount did not become ready")

    def _try_unmount_bcachefs_candidate(self) -> str | None:
        if self.candidate_mountpoint is None:
            return None
        mountpoint = self.candidate_mountpoint
        result = _run(["umount", mountpoint], check=False)
        if result.returncode:
            return f"candidate umount failed ({result.returncode}): {result.stdout}"
        self.candidate_mountpoint = None
        assert self.candidate_process is not None
        try:
            self.candidate_process.wait(timeout=CANDIDATE_MOUNT_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            self.candidate_process.terminate()
            self.candidate_process.wait(timeout=CANDIDATE_MOUNT_TIMEOUT_SECONDS)
            return "candidate FUSE process did not exit after unmount"
        finally:
            self.candidate_process = None
        try:
            mountpoint.rmdir()
        except OSError as error:
            return f"candidate mountpoint cleanup failed: {error}"
        return None

    def unmount_bcachefs_candidate(self) -> None:
        if error := self._try_unmount_bcachefs_candidate():
            pytest.fail(error)

    def cleanup(self) -> None:
        errors = [
            error
            for error in (
                self._try_unmount_bcachefs_candidate(),
                self._try_unmount(),
                self._try_detach_loop(),
            )
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


def _write_non_directory_entries(image: BtrfsImage) -> None:
    image.assert_writable_top_level()
    os.symlink("dir-0000/file-0000", image.mountpoint / "tree-link")
    os.mkfifo(image.mountpoint / "named-pipe")


COPY_PRESSURE_FILE_BYTES = 672 * 1024 * 1024


def _write_copy_pressure_fixture(image: BtrfsImage) -> None:
    image.assert_writable_top_level()
    source = image.mountpoint / "shared-source"
    chunk = b"\0" * (4 * 1024 * 1024)
    with source.open("wb") as file:
        for _ in range(COPY_PRESSURE_FILE_BYTES // len(chunk)):
            file.write(chunk)
        file.flush()
        os.fsync(file.fileno())
    _run(["cp", "--reflink=always", source, image.mountpoint / "shared-clone"])
    image.sync()


def _write_reflink_fragmented_fixture(image: BtrfsImage) -> None:
    image.assert_writable_top_level()
    block = bytes(range(256)) * (4096 // 256)

    (image.mountpoint / "inline-tail").write_bytes(b"tail")
    sparse = image.mountpoint / "sparse"
    with sparse.open("wb") as file:
        file.write(block)
        file.seek(4 * 1024 * 1024)
        file.write(block)
        file.flush()
        os.fsync(file.fileno())
    with (image.mountpoint / "unwritten").open("wb") as file:
        os.posix_fallocate(file.fileno(), 0, 4 * 1024 * 1024)
        file.flush()
        os.fsync(file.fileno())

    shared = image.mountpoint / "shared"
    with shared.open("wb") as file:
        for _ in range(256):
            file.write(block)
        file.flush()
        os.fsync(file.fileno())
    _run(["cp", "--reflink=always", shared, image.mountpoint / "shared-clone"])

    hardlinked = image.mountpoint / "hardlinked"
    hardlinked.write_bytes(block * 2)
    os.link(hardlinked, image.mountpoint / "hardlinked-alias")

    fragments = []
    for index in range(24):
        fragment = image.mountpoint / f"fragment-reserve-{index:02d}"
        fragment.write_bytes(block * 64)
        fragments.append(fragment)
    for fragment in fragments[::2]:
        fragment.unlink()

    with (image.mountpoint / "fragmented").open("wb") as file:
        for _ in range(128):
            file.write(block)
        file.flush()
        os.fsync(file.fileno())
    for fragment in fragments[1::2]:
        fragment.unlink()

    chunk = block * 1024
    with (image.mountpoint / "chunk-boundary").open("wb") as file:
        for _ in range(300):
            file.write(chunk)
        file.flush()
        os.fsync(file.fileno())
    image.sync()
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


def _start_migrate(
    image: BtrfsImage, *args: str, env: dict[str, str]
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [BCACHEFS, "migrate", *args, image.mountpoint],
        text=True,
        stdout=subprocess.PIPE,
        bufsize=1,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )


def _stop_migrate(process: subprocess.Popen[str]) -> str:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)

    try:
        output, _ = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate(timeout=5)

    return output

def _wait_for_migrate_exit(process: subprocess.Popen[str], timeout: float) -> str:
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        output = _stop_migrate(process)
        pytest.fail(f"migrate teardown exceeded {timeout}s:\n{output}")
    return output

def _wait_for_migrate_self_termination(
    process: subprocess.Popen[str], timeout: float
) -> str:
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        output = error.output or ""
        pytest.fail(
            f"migrate did not self-terminate within {timeout}s; "
            f"no signal was sent to process group {process.pid}:\n{output}"
        )
    return output




def _wait_for_migrate_output(
    process: subprocess.Popen[str], marker: str, timeout: float
) -> str:
    assert process.stdout is not None
    output = ""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ready, _, _ = select.select(
            [process.stdout], [], [], deadline - time.monotonic()
        )
        if not ready:
            break
        chunk = os.read(process.stdout.fileno(), 4096).decode()
        if not chunk:
            break
        output += chunk
        if marker in output:
            return output
    output += _stop_migrate(process)
    pytest.fail(f"migrate did not emit {marker!r}:\n{output}")


def _abort_diag(output: str) -> dict[str, str]:
    match = re.search(r"(?m)^MIGRATE_ABORT_DIAG (?P<fields>.+)$", output)
    assert match, output
    return dict(field.split("=", 1) for field in match["fields"].split())


def _wait_for_reservation(image: BtrfsImage, process: subprocess.Popen[str]) -> ReservationState:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if (image.mountpoint / "bcachefs").is_file():
            return _reservation_state(image)
        if process.poll() is not None:
            break
        time.sleep(0.05)
    output = _stop_migrate(process)
    pytest.fail(f"migrate exited before creating its Btrfs reservation:\n{output}")

@dataclass(frozen=True)
class ReservationState:
    inode: int
    size: int
    allocated_bytes: int


def _manifest(root: Path) -> dict[str, str]:
    manifest = {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in root.rglob("*")
        if path.is_file()
    }
    assert manifest, root
    return manifest


def _assert_manifest(root: Path, manifest: dict[str, str]) -> None:
    for relative_path, expected_digest in manifest.items():
        path = root / relative_path
        assert path.is_file(), path
        assert _sha256(path) == expected_digest, path


def _reservation_state(image: BtrfsImage) -> ReservationState:
    reservation = image.mountpoint / "bcachefs"
    stat = reservation.stat()
    assert reservation.is_file(), reservation
    assert stat.st_size > 0, stat
    allocated_bytes = stat.st_blocks * 512
    assert allocated_bytes > 0, stat
    return ReservationState(stat.st_ino, stat.st_size, allocated_bytes)


def _alternate_superblock(output: str) -> int:
    match = ALTERNATE_SUPERBLOCK_RE.search(output)
    assert match, output
    return int(match.group(1))


def _assert_successful_migration(
    result: subprocess.CompletedProcess[str],
) -> tuple[dict[str, int], int]:
    assert result.returncode == 0, result.stdout
    assert "MIGRATE_RESULT eligible=yes" in result.stdout, result.stdout
    assert re.search(r"(?m)^MIGRATE_RESERVATION ", result.stdout), result.stdout
    _assert_bounded_allocator_result(result)
    return _reservation_metrics(result.stdout), _alternate_superblock(result.stdout)


def _assert_device_stats_zero(image: BtrfsImage) -> None:
    stats = _run(["btrfs", "device", "stats", image.mountpoint]).stdout
    values = [int(value) for value in re.findall(r"(?m)\s(\d+)$", stats)]
    assert values and not any(values), stats


def test_a_supported_btrfs_no_change_is_eligible(btrfs_image: BtrfsImage) -> None:
    _write_tree(btrfs_image, directories=4, files_per_directory=8)
    btrfs_image.sync()
    btrfs_image.switch_to_readonly_top_level()
    btrfs_image.assert_readonly_top_level()
    image_before = _sha256(btrfs_image.image)

    result = _migrate(btrfs_image, "--no-change", "-f")

    assert result.returncode == 0, result.stdout
    assert "MIGRATE_RESULT eligible=yes" in result.stdout, result.stdout
    _copy_reasons(result.stdout, "no-change")
    assert not (btrfs_image.mountpoint / "bcachefs").exists()
    assert _sha256(btrfs_image.image) == image_before


def test_b_supported_btrfs_migrates_with_full_reservation_metrics(
    btrfs_image: BtrfsImage,
) -> None:
    _write_tree(btrfs_image, directories=8, files_per_directory=16)
    _write_non_directory_entries(btrfs_image)
    btrfs_image.sync()

    result = _migrate(btrfs_image, "-f")

    assert result.returncode == 0, result.stdout
    assert "MIGRATE_RESULT eligible=yes" in result.stdout, result.stdout
    _assert_bounded_allocator_result(result)
    metrics = _reservation_metrics(result.stdout)
    assert metrics["reservation_usable_full_bucket_bytes"] > 0
    nouse = _reservation_nouse_metrics(result.stdout)
    assert (
        nouse["reservation_allocatable_bytes"]
        == metrics["reservation_usable_full_bucket_bytes"]
    )
    progress = _migration_progress(result.stdout)
    assert len(progress) == 2, result.stdout
    assert progress[0]["entries_done"] == 0
    assert progress[-1]["entries_done"] == progress[-1]["entries_total"] == 138
    assert progress[-1]["non_directories_done"] == 130
    assert progress[-1]["directories_done"] == 8
    assert (
        progress[-1]["non_directories_done"] + progress[-1]["directories_done"]
        == progress[-1]["entries_total"]
    )
    _copy_reasons(result.stdout, "migrate")
    assert (btrfs_image.mountpoint / "bcachefs").is_file()
    candidate = btrfs_image.mount_bcachefs_candidate_readonly(
        _alternate_superblock(result.stdout)
    )
    try:
        migrated_link = candidate / "tree-link"
        assert migrated_link.is_symlink()
        assert os.readlink(migrated_link) == "dir-0000/file-0000"
        assert stat.S_ISFIFO((candidate / "named-pipe").stat().st_mode)
    finally:
        btrfs_image.unmount_bcachefs_candidate()
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



def test_key_cache_going_ro_split_exhaustion_shutdown_is_bounded(
    btrfs_image: BtrfsImage,
) -> None:
    _write_tree(btrfs_image, directories=4, files_per_directory=8)
    btrfs_image.sync()
    manifest = _manifest(btrfs_image.mountpoint)
    process = _start_migrate(
        btrfs_image,
        "-f",
        env={
            **os.environ,
            "BCACHEFS_TEST_GOING_RO_ABORT_REPRODUCER": "1",
        },
    )
    reservation_before = _wait_for_reservation(btrfs_image, process)
    output = _wait_for_migrate_output(
        process, "MIGRATE_ABORT_REPRODUCER_READY ", timeout=30
    )
    output += _wait_for_migrate_exit(process, GOING_RO_ABORT_TIMEOUT_SECONDS)

    assert process.returncode != 0, output

    assert _reservation_state(btrfs_image) == reservation_before
    _assert_manifest(btrfs_image.mountpoint, manifest)
    diag = _abort_diag(output)
    assert diag["classification"] == "nouse_exhausted", output
    assert diag["allocator_outcome"] == "no_usable_buckets", output
    assert int(diag["key_cache_nr_dirty"]) > 0, output
    assert int(diag["reserve_cache_nr"]) >= 0, output
    assert int(diag["open_buckets_nr_free"]) >= 0, output
    assert diag["watermark"], output
    assert int(diag["reserve_get_ret"]) != 0, output
    assert int(diag["complete_device_scans"]) > 0, output
    assert int(diag["split_attempts"]) > 0, output
    assert int(diag["buckets_seen"]) > 0, output
    assert int(diag["skipped_open"]) >= 0, output
    assert int(diag["skipped_need_journal_commit"]) >= 0, output
    assert int(diag["skipped_nouse"]) > 0, output
    assert int(diag["skipped_mi_btree_bitmap"]) >= 0, output
    assert int(diag["trace_entries"]) > 0, output
    assert "MIGRATE_ABORT_TRACE " in output, output
    assert "MIGRATE_ABORT_KEYCACHE result=" in output, output
    assert "MIGRATE_ABORT_RO_STAGE clean_loop_complete" in output, output
    assert "MIGRATE_ABORT_RO_STAGE writes_disabled_finally" in output, output

def test_sigint_after_dirty_metadata_cancels_at_safe_boundary(
    btrfs_image: BtrfsImage,
) -> None:
    _write_tree(btrfs_image, directories=4, files_per_directory=8)
    btrfs_image.sync()
    manifest = _manifest(btrfs_image.mountpoint)
    process = _start_migrate(
        btrfs_image,
        "-f",
        env={
            **os.environ,
            "BCACHEFS_TEST_INTERRUPT_AFTER_DIRTY": "1",
        },
    )
    reservation_before = _wait_for_reservation(btrfs_image, process)
    output = _wait_for_migrate_output(process, "MIGRATE_INTERRUPT_READY", timeout=30)
    os.killpg(process.pid, signal.SIGINT)
    output += _wait_for_migrate_exit(process, GOING_RO_ABORT_TIMEOUT_SECONDS)

    assert process.returncode != 0, output
    assert "MIGRATE_CANCELLATION requested=true" in output, output
    assert "MIGRATE_ABORT_REPRODUCER_READY key_cache_nr_dirty=" in output, output
    assert "MIGRATE_ABORT_LOOP" not in output, output
    assert "Migrate complete" not in output, output
    assert _reservation_state(btrfs_image) == reservation_before
    _assert_manifest(btrfs_image.mountpoint, manifest)


def test_stress_migrate_is_bounded_and_avoids_allocator_path_leaks(
    btrfs_image: BtrfsImage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fragment_available_space(btrfs_image)
    # This remains small enough for a routine integration run while exercising
    # the actual allocator scan. The binary reports per-scan path high-water
    # marks only when this fixture enables the test-only metrics switch.
    _write_tree(btrfs_image, directories=128, files_per_directory=48)
    btrfs_image.sync()
    btrfs_image.assert_writable_top_level()
    monkeypatch.setenv("BCACHEFS_TEST_ALLOCATOR_SCAN_METRICS", "1")

    result = _migrate(btrfs_image, "-f")

    assert result.returncode == 0, result.stdout
    _assert_bounded_allocator_result(result)
    metrics = _reservation_metrics(result.stdout)
    scan_metrics = _allocator_scan_metrics(result.stdout)
    assert metrics["reservation_usable_full_bucket_bytes"] > 0, result.stdout
    assert scan_metrics["scans"] > 0, result.stdout
    assert scan_metrics["zero_ref_preserved_freespace_paths"] <= 1, result.stdout
    assert scan_metrics["max_paths"] <= 16, result.stdout
    assert "MIGRATE_RESULT eligible=yes" in result.stdout, result.stdout

def test_allocator_scan_restart_unwinds_and_retries(
    btrfs_image: BtrfsImage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fragment_available_space(btrfs_image)
    _write_tree(btrfs_image, directories=32, files_per_directory=16)
    btrfs_image.sync()
    btrfs_image.assert_writable_top_level()
    monkeypatch.setenv("BCACHEFS_TEST_ALLOCATOR_SCAN_METRICS", "1")
    monkeypatch.setenv("BCACHEFS_TEST_ALLOCATOR_SCAN_INJECT_RESTART", "1")

    result = _migrate(btrfs_image, "-f")

    assert result.returncode == 0, result.stdout
    _assert_bounded_allocator_result(result)
    scan_metrics = _allocator_scan_metrics(result.stdout)
    assert scan_metrics["injected_restarts"] == 1, result.stdout
    assert scan_metrics["scans"] > 1, result.stdout
    assert scan_metrics["zero_ref_preserved_freespace_paths"] <= 1, result.stdout
    assert scan_metrics["max_paths"] <= 16, result.stdout



def test_copy_capacity_preflight_rejects_before_candidate_writes(
    btrfs_image: BtrfsImage,
) -> None:
    # Populate foreground metadata before the shared-data copies consume the
    # candidate reservation: the later exhaustion must flush that normal
    # foreground write buffer, not a synthetic post-copy reproducer.
    _write_tree(btrfs_image, directories=128, files_per_directory=48)
    _write_copy_pressure_fixture(btrfs_image)
    manifest = _manifest(btrfs_image.mountpoint)

    no_change = _migrate(btrfs_image, "--no-change", "-f")
    assert no_change.returncode == 0, no_change.stdout
    no_change_reasons = _copy_reasons(no_change.stdout, "no-change")
    assert no_change_reasons["copied_shared_bytes"] >= 2 * COPY_PRESSURE_FILE_BYTES

    result = _migrate(btrfs_image, "-f")

    assert result.returncode != 0, result.stdout
    assert "MIGRATE_COPY_FEASIBILITY verdict=reject" in result.stdout, result.stdout
    assert "initializing new filesystem" not in result.stdout, result.stdout
    assert "Migrate complete" not in result.stdout, result.stdout
    reservation_before = _reservation_state(btrfs_image)
    _assert_manifest(btrfs_image.mountpoint, manifest)

    legacy = _start_migrate(
        btrfs_image,
        "-F",
        "-f",
        env={
            **os.environ,
            "BCACHEFS_TEST_BYPASS_COPY_FEASIBILITY": "1",
            "BCACHEFS_TEST_MIGRATE_STOP_TRACE": "1",
        },
    )
    legacy_output = _wait_for_migrate_self_termination(
        legacy, GOING_RO_ABORT_TIMEOUT_SECONDS
    )
    assert legacy.returncode != 0, legacy_output
    assert "MIGRATE_RESERVATION_NOUSE " in legacy_output, legacy_output
    assert "no_buckets_found" in legacy_output, legacy_output
    assert (
        "bch2_btree_write_buffer_flush_locked(): fatal error no_buckets_found"
        in legacy_output
    ), legacy_output
    for marker in (
        "MIGRATE_STOP stage=fatal_error",
        "MIGRATE_STOP stage=emergency_ro",
        "MIGRATE_STOP stage=copy_return result=error",
        "MIGRATE_STOP stage=fs_exit_begin",
        "MIGRATE_STOP stage=fs_stop_enter",
        "MIGRATE_STOP stage=write_buffer_flush_begin",
        "MIGRATE_STOP stage=emergency_write_cancel_complete",
        "MIGRATE_STOP stage=fs_stop_return",
        "MIGRATE_STOP stage=fs_exit_complete result=error",
    ):
        assert marker in legacy_output, legacy_output
    assert legacy_output.index("MIGRATE_STOP stage=fatal_error") < legacy_output.index(
        "MIGRATE_STOP stage=emergency_ro"
    )
    assert legacy_output.index(
        "MIGRATE_STOP stage=copy_return result=error"
    ) < legacy_output.index("MIGRATE_STOP stage=fs_exit_begin")
    assert legacy_output.index(
        "MIGRATE_STOP stage=fs_stop_enter"
    ) < legacy_output.index("MIGRATE_STOP stage=fs_stop_return")
    assert legacy_output.index(
        "MIGRATE_STOP stage=fs_stop_return"
    ) < legacy_output.index("MIGRATE_STOP stage=fs_exit_complete result=error")
    assert _reservation_state(btrfs_image) == reservation_before
    _assert_manifest(btrfs_image.mountpoint, manifest)

def test_retry_no_change_classifier_matches_runtime_with_existing_reservation(
    btrfs_image: BtrfsImage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_reflink_fragmented_fixture(btrfs_image)
    _write_tree(btrfs_image, directories=2, files_per_directory=4)
    _write_non_directory_entries(btrfs_image)
    btrfs_image.sync()
    monkeypatch.setenv("BCACHEFS_TEST_COPY_DIAGNOSTICS", "1")
    manifest = _manifest(btrfs_image.mountpoint)

    initial_plan = _migrate(btrfs_image, "--no-change", "-f")
    assert initial_plan.returncode == 0, initial_plan.stdout
    initial_reasons = _copy_reasons(initial_plan.stdout, "no-change")
    initial_stats = _migration_plan(initial_plan.stdout, "no-change")
    assert initial_reasons["copied_shared_bytes"] > 0
    assert initial_reasons["copied_inline_bytes"] > 0
    assert initial_reasons["copied_symlink_bytes"] > 0
    assert initial_stats["mapped_extent_count"] > 1
    assert int(re.search(r"data_chunk_count=(\d+)", initial_plan.stdout).group(1)) > 1

    process = _start_migrate(
        btrfs_image,
        "-f",
        env={
            **os.environ,
            "BCACHEFS_TEST_INTERRUPT_AFTER_DIRTY": "1",
        },
    )
    reservation_before = _wait_for_reservation(btrfs_image, process)
    interrupted = _wait_for_migrate_output(process, "MIGRATE_INTERRUPT_READY", timeout=30)
    os.killpg(process.pid, signal.SIGINT)
    interrupted += _wait_for_migrate_exit(process, GOING_RO_ABORT_TIMEOUT_SECONDS)
    assert process.returncode != 0, interrupted
    assert _reservation_state(btrfs_image) == reservation_before
    _assert_manifest(btrfs_image.mountpoint, manifest)

    retry_plan = _migrate(btrfs_image, "--no-change", "-F", "-f")
    assert retry_plan.returncode == 0, retry_plan.stdout
    retry_reasons = _copy_reasons(retry_plan.stdout, "no-change")
    retry_stats = _migration_plan(retry_plan.stdout, "no-change")
    assert retry_reasons == initial_reasons
    assert retry_stats == initial_stats

    retried = _migrate(btrfs_image, "-F", "-f")
    assert retried.returncode == 0, retried.stdout
    assert _copy_reasons(retried.stdout, "preflight") == retry_reasons
    assert _copy_reasons(retried.stdout, "migrate") == retry_reasons
    assert _migration_plan(retried.stdout, "migrate") == retry_stats
    assert _copy_policy(retry_plan.stdout, "no-change") == _copy_policy(
        retried.stdout, "migrate"
    )
    top_reasons = re.findall(
        r"(?m)^MIGRATE_COPY_TOP mode=migrate rank=(\d+) reason=(\w+) copied_bytes=(\d+)$",
        retried.stdout,
    )
    assert top_reasons, retried.stdout
    assert top_reasons[0][1] == "shared"
    assert _reservation_state(btrfs_image).inode == reservation_before.inode
    _assert_manifest(btrfs_image.mountpoint, manifest)

    candidate = btrfs_image.mount_bcachefs_candidate_readonly(
        _alternate_superblock(retried.stdout)
    )
    try:
        _assert_manifest(candidate, manifest)
    finally:
        btrfs_image.unmount_bcachefs_candidate()
def test_retry_existing_migration_reservation_is_bounded_and_valid(
    btrfs_image: BtrfsImage,
) -> None:
    _fragment_available_space(btrfs_image)
    _write_tree(btrfs_image, directories=64, files_per_directory=32)
    btrfs_image.sync()
    manifest = _manifest(btrfs_image.mountpoint)

    first_metrics, first_sb = _assert_successful_migration(_migrate(btrfs_image, "-f"))
    first_reservation = _reservation_state(btrfs_image)

    first_candidate = btrfs_image.mount_bcachefs_candidate_readonly(first_sb)
    try:
        assert (first_candidate / "old_migrated_filesystem").exists()
        _assert_manifest(first_candidate, manifest)
    finally:
        btrfs_image.unmount_bcachefs_candidate()

    btrfs_image.mount_readwrite_top_level()
    btrfs_image.assert_writable_top_level()
    assert (btrfs_image.mountpoint / "bcachefs").is_file()
    _assert_manifest(btrfs_image.mountpoint, manifest)

    second_metrics, second_sb = _assert_successful_migration(
        _migrate(btrfs_image, "-F", "-f")
    )
    second_reservation = _reservation_state(btrfs_image)
    assert second_reservation.inode == first_reservation.inode
    assert second_reservation.size == first_reservation.size
    assert second_reservation.allocated_bytes >= first_reservation.allocated_bytes

    second_candidate = btrfs_image.mount_bcachefs_candidate_readonly(second_sb)
    try:
        assert (second_candidate / "old_migrated_filesystem").exists()
        _assert_manifest(second_candidate, manifest)
    finally:
        btrfs_image.unmount_bcachefs_candidate()

    btrfs_image.mount("ro,rescue=nologreplay,noatime,subvolid=5")
    btrfs_image.assert_readonly_top_level()
    assert (btrfs_image.mountpoint / "bcachefs").is_file()
    _assert_manifest(btrfs_image.mountpoint, manifest)
    _assert_device_stats_zero(btrfs_image)
    btrfs_image.check_readonly()

    print(
        "MIGRATE_RETRY_FIRST "
        f"sb={first_sb} inode={first_reservation.inode} "
        f"size={first_reservation.size} allocated={first_reservation.allocated_bytes} "
        f"raw_extents={first_metrics['reservation_raw_extent_count']} "
        f"raw_bytes={first_metrics['reservation_raw_bytes']} "
        f"bucket_bytes={first_metrics['reservation_bucket_size_bytes']} "
        f"full_buckets={first_metrics['reservation_usable_full_bucket_count']} "
        f"full_bytes={first_metrics['reservation_usable_full_bucket_bytes']} "
        f"alignment_loss={first_metrics['reservation_bucket_alignment_loss_bytes']} "
        f"superblock_protected={first_metrics['reservation_default_superblock_protected_bytes']} "
        f"device_tail={first_metrics['reservation_device_tail_bytes']}"
    )
    print(
        "MIGRATE_RETRY_SECOND "
        f"sb={second_sb} inode={second_reservation.inode} "
        f"size={second_reservation.size} allocated={second_reservation.allocated_bytes} "
        f"raw_extents={second_metrics['reservation_raw_extent_count']} "
        f"raw_bytes={second_metrics['reservation_raw_bytes']} "
        f"bucket_bytes={second_metrics['reservation_bucket_size_bytes']} "
        f"full_buckets={second_metrics['reservation_usable_full_bucket_count']} "
        f"full_bytes={second_metrics['reservation_usable_full_bucket_bytes']} "
        f"alignment_loss={second_metrics['reservation_bucket_alignment_loss_bytes']} "
        f"superblock_protected={second_metrics['reservation_default_superblock_protected_bytes']} "
        f"device_tail={second_metrics['reservation_device_tail_bytes']}"
    )
