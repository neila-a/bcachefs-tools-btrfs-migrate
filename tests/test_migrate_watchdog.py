from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WATCHDOG = ROOT / "scripts" / "migrate-watchdog.py"


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
