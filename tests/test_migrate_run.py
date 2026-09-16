import importlib.util
import json
import os
import stat
import struct
import sys
import uuid

from types import SimpleNamespace

from pathlib import Path

import pytest


RUNNER = Path(__file__).parents[1] / "scripts" / "migrate-run.py"


def _load_runner_module() -> object:
    spec = importlib.util.spec_from_file_location("migrate_run_under_test", RUNNER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_mountinfo_detects_btrfs_without_using_mount_device_number() -> None:
    module = _load_runner_module()
    mountinfo = (
        "42 35 0:84 / /mnt/source rw,relatime - btrfs /dev/nvme0n1p4 rw\n"
        "43 35 259:4 / /mnt/other rw,relatime - ext4 /dev/nvme0n1p4 rw\n"
    )

    mounts = module.parse_btrfs_mountinfo(mountinfo)

    assert [(mount.mount_point, mount.source) for mount in mounts] == [
        ("/mnt/source", "/dev/nvme0n1p4")
    ]


def test_pytest_loop_mountinfo_is_refused_before_ioctl_open() -> None:
    module = _load_runner_module()
    mounts = module.parse_btrfs_mountinfo(
        "42 35 0:84 / /tmp/pytest-of-root/pytest-12/test_case/mnt rw - "
        "btrfs /dev/loop7 rw\n"
    )

    with pytest.raises(module.UnsafeTestMountError, match="pytest-of-root"):
        module.ensure_no_test_btrfs_mountinfo("mnt:[4026531840]", mounts)


def test_fs_info_decodes_the_ioctl_fsid_from_a_mutable_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_runner_module()
    expected_fsid = uuid.UUID("01234567-89ab-cdef-0123-456789abcdef")

    def fake_ioctl(_fd: int, request: int, payload: bytearray, _mutate: bool) -> None:
        assert request == module.BTRFS_IOC_FS_INFO
        struct.pack_into("Q", payload, 0, 7)
        payload[16:32] = expected_fsid.bytes

    monkeypatch.setattr(module.fcntl, "ioctl", fake_ioctl)

    assert module._fs_info(42) == (7, str(expected_fsid))


def test_snapshot_is_verified_and_isolated_from_the_build_artifact(
    tmp_path: Path,
) -> None:
    module = _load_runner_module()
    source = tmp_path / "bcachefs-build-output"
    source.write_text("#!/bin/sh\nprintf 'fixture-v1\\n'\n")
    source.chmod(0o755)
    run_dir = tmp_path / "run"

    snapshot, manifest = module.create_run_snapshot(source, run_dir)
    source.write_text("#!/bin/sh\nprintf 'fixture-v2\\n'\n")

    assert snapshot == run_dir / "binary" / "bcachefs"
    assert snapshot.read_text() == "#!/bin/sh\nprintf 'fixture-v1\\n'\n"
    assert stat.S_IMODE(snapshot.stat().st_mode) == 0o500
    assert manifest["binary"]["sha256"] == module.sha256_file(snapshot)
    assert manifest["binary"]["version"] == "fixture-v1"
    assert manifest["binary"]["source"] == str(source)
    assert json.loads((run_dir / "manifest.json").read_text()) == manifest


def test_pytest_or_loop_backed_btrfs_mounts_are_refused() -> None:
    module = _load_runner_module()
    test_mount = module.BtrfsMount(
        namespace="mnt:[4026531840]",
        mount_point="/tmp/pytest-of-root/pytest-12/test_case/mnt",
        fsid="01234567-89ab-cdef-0123-456789abcdef",
        devices=("/dev/loop7",),
    )

    with pytest.raises(module.UnsafeTestMountError, match="pytest-of-root"):
        module.ensure_no_test_btrfs_mounts([test_mount])


def test_unrelated_real_btrfs_mount_is_not_rejected_as_a_test_fixture() -> None:
    module = _load_runner_module()
    real_mount = module.BtrfsMount(
        namespace="mnt:[4026531840]",
        mount_point="/srv/data",
        fsid="01234567-89ab-cdef-0123-456789abcdef",
        devices=("/dev/nvme0n1p4",),
    )

    module.ensure_no_test_btrfs_mounts([real_mount])


def test_main_passes_the_migrate_root_with_dash_f(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_runner_module()
    snapshot = tmp_path / "run" / "binary" / "bcachefs"
    launched: list[str] = []

    monkeypatch.setattr(module, "discover_btrfs_mounts", lambda: [])
    monkeypatch.setattr(module, "ensure_no_test_btrfs_mounts", lambda _mounts: None)
    monkeypatch.setattr(
        module,
        "create_run_snapshot",
        lambda _binary, _run_dir: (snapshot, {"binary": {"sha256": "fixture"}}),
    )
    monkeypatch.setattr(module, "verify_snapshot", lambda _snapshot, _manifest: None)

    def fake_run(command: list[str], check: bool) -> SimpleNamespace:
        assert check is False
        launched.extend(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    args = SimpleNamespace(
        source=Path("/fixture/source"),
        binary=Path("/fixture/bcachefs"),
        run_dir=tmp_path / "run",
        watchdog=Path("/fixture/migrate-watchdog.py"),
        migration_args=["--no-change", "-F"],
    )

    assert module.main(args) == 0
    assert launched == [
        sys.executable,
        "/fixture/migrate-watchdog.py",
        "--log-file",
        str(tmp_path / "run" / "migrate-watchdog.log"),
        "--",
        str(snapshot),
        "migrate",
        "-f",
        "/fixture/source",
        "--no-change",
        "-F",
    ]
