from __future__ import annotations

import importlib.util
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
WATCHDOG = ROOT / "scripts" / "migrate-watchdog.py"

def _watchdog_command(
    child: str,
    *,
    hard_timeout: str = "0.2",
    cancel_grace: str = "0.1",
    term_grace: str = "0.1",
    log_file: Path | None = None,
) -> list[str | Path]:
    command: list[str | Path] = [
        sys.executable,
        WATCHDOG,
        "--hard-timeout-seconds",
        hard_timeout,
        "--stall-timeout-seconds",
        "10",
        "--cancel-grace-seconds",
        cancel_grace,
        "--term-grace-seconds",
        term_grace,
    ]
    if log_file is not None:
        command.extend(("--log-file", log_file))
    command.extend(("--", sys.executable, "-c", child))
    return command


def _wait_for_pid(path: Path, timeout: float = 5) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return int(path.read_text())
        time.sleep(0.01)
    pytest.fail(f"child did not record its PID at {path}")


def _pid_is_gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _wait_for_pid_exit(pid: int, timeout: float = 5) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _pid_is_gone(pid):
            return True
        time.sleep(0.01)
    return _pid_is_gone(pid)


def _process_group_is_gone(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _cleanup_process_group(process: subprocess.Popen[str], pgid: int | None) -> None:
    if process.poll() is None:
        process.kill()
        process.wait(timeout=5)
    if pgid is not None and not _process_group_is_gone(pgid):
        os.killpg(pgid, signal.SIGKILL)


def _load_watchdog_module() -> object:
    name = f"migrate_watchdog_test_{time.monotonic_ns()}"
    spec = importlib.util.spec_from_file_location(name, WATCHDOG)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _progress(entries_done: int) -> str:
    return (
        "MIGRATE_PROGRESS "
        f"entries_done={entries_done} entries_total=2 "
        f"non_directories_done={entries_done} directories_done=0 "
        f"linked_bytes={entries_done} copied_bytes=0 "
        f"elapsed_seconds={entries_done} capacity_bytes=2 "
        f"used_bytes={entries_done} free_bytes={2 - entries_done}"
    )


def test_hard_ceiling_captures_backtrace_and_never_reports_success_after_sigkill() -> None:
    child = (
        "import signal, time; "
        "signal.signal(signal.SIGINT, signal.SIG_IGN); "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('CHILD_STARTED', flush=True); "
        "time.sleep(60)"
    )
    result = subprocess.run(
        [
            sys.executable,
            WATCHDOG,
            "--hard-timeout-seconds",
            "0.2",
            "--stall-timeout-seconds",
            "10",
            "--cancel-grace-seconds",
            "0.2",
            "--term-grace-seconds",
            "0.2",
            "--",
            sys.executable,
            "-c",
            child,
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=15,
    )

    assert result.returncode == 137, result.stdout
    assert "CHILD_STARTED" in result.stdout
    assert "MIGRATE_WATCHDOG stage=hard_timeout" in result.stdout
    assert "MIGRATE_WATCHDOG stage=graceful_cancel reason=hard_timeout signal=SIGINT" in result.stdout
    assert "MIGRATE_WATCHDOG_BACKTRACE " in result.stdout
    assert "MIGRATE_WATCHDOG stage=sigterm" in result.stdout
    assert "MIGRATE_WATCHDOG stage=sigkill" in result.stdout
    assert "MIGRATE_WATCHDOG result=aborted forced=true" in result.stdout
    assert "MIGRATE_WATCHDOG result=success" not in result.stdout


def test_stall_backtrace_precedes_hard_ceiling_without_terminating_child() -> None:
    child = (
        "import signal, time; "
        "signal.signal(signal.SIGINT, signal.SIG_IGN); "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"print({_progress(0)!r}, flush=True); "
        "time.sleep(60)"
    )
    result = subprocess.run(
        [
            sys.executable,
            WATCHDOG,
            "--hard-timeout-seconds",
            "0.5",
            "--stall-timeout-seconds",
            "0.1",
            "--cancel-grace-seconds",
            "0.1",
            "--term-grace-seconds",
            "0.1",
            "--",
            sys.executable,
            "-c",
            child,
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=15,
    )

    assert result.returncode == 137, result.stdout
    assert "MIGRATE_WATCHDOG stage=stall" in result.stdout
    assert "MIGRATE_WATCHDOG stage=hard_timeout" in result.stdout
    assert result.stdout.index("stage=stall") < result.stdout.index("stage=hard_timeout")


def test_progress_advancement_resets_stall_timer() -> None:
    child = (
        "import time; "
        f"print({_progress(0)!r}, flush=True); "
        "time.sleep(0.12); "
        f"print({_progress(1)!r}, flush=True); "
        "time.sleep(0.12)"
    )
    result = subprocess.run(
        [
            sys.executable,
            WATCHDOG,
            "--hard-timeout-seconds",
            "1",
            "--stall-timeout-seconds",
            "0.2",
            "--cancel-grace-seconds",
            "0.1",
            "--term-grace-seconds",
            "0.1",
            "--",
            sys.executable,
            "-c",
            child,
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=15,
    )

    assert result.returncode == 0, result.stdout
    assert "MIGRATE_WATCHDOG stage=stall" not in result.stdout
    assert "MIGRATE_WATCHDOG result=success" in result.stdout


def test_external_cancellation_requests_sigint_before_sigterm() -> None:
    child = (
        "import signal, sys, time; "
        "signal.signal(signal.SIGINT, lambda *_: print('CHILD_SIGINT', flush=True)); "
        "signal.signal(signal.SIGTERM, lambda *_: (print('CHILD_SIGTERM', flush=True), sys.exit(0))); "
        "print('CHILD_STARTED', flush=True); "
        "time.sleep(60)"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            WATCHDOG,
            "--hard-timeout-seconds",
            "30",
            "--stall-timeout-seconds",
            "30",
            "--cancel-grace-seconds",
            "0.1",
            "--term-grace-seconds",
            "0.1",
            "--",
            sys.executable,
            "-c",
            child,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert process.stdout is not None
    assert process.stdout.readline() == "CHILD_STARTED\n"
    os.kill(process.pid, signal.SIGTERM)
    output, _ = process.communicate(timeout=15)

    assert process.returncode == 124, output
    assert "MIGRATE_WATCHDOG stage=cancel_requested signal=SIGTERM" in output
    assert "MIGRATE_WATCHDOG stage=graceful_cancel reason=external signal=SIGINT" in output
    assert "CHILD_SIGINT" in output
    assert "MIGRATE_WATCHDOG_BACKTRACE " in output
    assert "MIGRATE_WATCHDOG stage=sigterm" in output
    assert "CHILD_SIGTERM" in output
    assert "MIGRATE_WATCHDOG result=aborted forced=false" in output


def test_fatal_child_output_that_stays_alive_is_reaped_after_hard_timeout(
    tmp_path: Path,
) -> None:
    pid_path = tmp_path / "child.pid"
    child = (
        "import os, signal, time; from pathlib import Path; "
        f"Path({str(pid_path)!r}).write_text(str(os.getpid())); "
        "signal.signal(signal.SIGINT, signal.SIG_IGN); "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('CHILD_FATAL no_buckets_found', flush=True); "
        "time.sleep(60)"
    )

    result = subprocess.run(
        _watchdog_command(child),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=15,
    )

    pid = _wait_for_pid(pid_path)
    assert result.returncode == 137, result.stdout
    assert "CHILD_FATAL no_buckets_found" in result.stdout
    assert _wait_for_pid_exit(pid), result.stdout


def test_internal_watchdog_exception_reaps_the_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_watchdog_module()
    pid_path = tmp_path / "child.pid"
    child = (
        "import os, signal, time; from pathlib import Path; "
        f"Path({str(pid_path)!r}).write_text(str(os.getpid())); "
        "signal.signal(signal.SIGINT, signal.SIG_IGN); "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(60)"
    )
    seen: list[subprocess.Popen[str]] = []

    def explode(_monitor, process, _now, _timeout) -> None:
        seen.append(process)
        _wait_for_pid(pid_path)
        raise RuntimeError("test supervisor failure")

    monkeypatch.setattr(module.ProgressMonitor, "capture_stall_if_needed", explode)
    args = SimpleNamespace(
        command=[sys.executable, "-c", child],
        hard_timeout_seconds=30,
        stall_timeout_seconds=30,
        cancel_grace_seconds=0.1,
        term_grace_seconds=0.1,
        log_file=None,
    )
    pgid: int | None = None
    try:
        with pytest.raises(RuntimeError, match="test supervisor failure"):
            module.run(args)
        assert seen
        pid = _wait_for_pid(pid_path)
        pgid = seen[0].pid
        assert _wait_for_pid_exit(pid)
        assert _process_group_is_gone(pgid)
    finally:
        if seen:
            process = seen[0]
            _cleanup_process_group(process, pgid)


def test_broken_stdout_keeps_a_file_log_and_reaps_the_child(tmp_path: Path) -> None:
    pid_path = tmp_path / "child.pid"
    log_file = tmp_path / "watchdog.log"
    child = (
        "import os, signal, time; from pathlib import Path; "
        f"Path({str(pid_path)!r}).write_text(str(os.getpid())); "
        "signal.signal(signal.SIGINT, signal.SIG_IGN); "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "exec(\"while True:\\n print('CHILD_FATAL no_buckets_found', flush=True); time.sleep(0.01)\")"
    )
    process = subprocess.Popen(
        _watchdog_command(child, log_file=log_file),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert process.stdout is not None
    process.stdout.close()
    pgid: int | None = None
    try:
        pid = _wait_for_pid(pid_path)
        pgid = os.getpgid(pid)
        process.wait(timeout=15)
        assert process.returncode != 0
        assert _wait_for_pid_exit(pid)
        assert _process_group_is_gone(pgid)
        output = log_file.read_text()
        assert "CHILD_FATAL no_buckets_found" in output
        assert "MIGRATE_WATCHDOG stage=stdout_unavailable" in output
    finally:
        _cleanup_process_group(process, pgid)


def test_watchdog_sigterm_reaps_the_entire_child_process_group(tmp_path: Path) -> None:
    parent_pid_path = tmp_path / "parent.pid"
    grandchild_pid_path = tmp_path / "grandchild.pid"
    grandchild = (
        "import os, signal, time; from pathlib import Path; "
        f"Path({str(grandchild_pid_path)!r}).write_text(str(os.getpid())); "
        "signal.signal(signal.SIGINT, signal.SIG_IGN); "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(60)"
    )
    child = (
        "import os, signal, subprocess, sys, time; from pathlib import Path; "
        f"Path({str(parent_pid_path)!r}).write_text(str(os.getpid())); "
        f"subprocess.Popen([sys.executable, '-c', {grandchild!r}]); "
        "signal.signal(signal.SIGINT, signal.SIG_IGN); "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('CHILD_STARTED', flush=True); "
        "time.sleep(60)"
    )
    process = subprocess.Popen(
        _watchdog_command(child, hard_timeout="30"),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    pgid: int | None = None
    try:
        parent_pid = _wait_for_pid(parent_pid_path)
        grandchild_pid = _wait_for_pid(grandchild_pid_path)
        pgid = os.getpgid(parent_pid)
        os.kill(process.pid, signal.SIGTERM)
        process.communicate(timeout=15)
        assert process.returncode == 137
        assert _wait_for_pid_exit(parent_pid)
        assert _wait_for_pid_exit(grandchild_pid)
        assert _process_group_is_gone(pgid)
    finally:
        _cleanup_process_group(process, pgid)


def test_watchdog_death_kills_its_child_process_group(tmp_path: Path) -> None:
    pid_path = tmp_path / "child.pid"
    child = (
        "import os, signal, time; from pathlib import Path; "
        f"Path({str(pid_path)!r}).write_text(str(os.getpid())); "
        "signal.signal(signal.SIGINT, signal.SIG_IGN); "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(60)"
    )
    process = subprocess.Popen(
        _watchdog_command(child, hard_timeout="30"),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    pgid: int | None = None
    try:
        pid = _wait_for_pid(pid_path)
        pgid = os.getpgid(pid)
        process.kill()
        process.wait(timeout=5)
        assert _wait_for_pid_exit(pid)
        assert _process_group_is_gone(pgid)
    finally:
        _cleanup_process_group(process, pgid)
