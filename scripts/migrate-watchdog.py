#!/usr/bin/env python3
"""Progress-aware supervisor for a bcachefs migration subprocess."""

from __future__ import annotations

import argparse
import ctypes
import errno
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import TextIO


# `non_directories_done` includes regular files, symlinks, and special files.
PROGRESS_FIELDS = (
    "entries_done",
    "entries_total",
    "non_directories_done",
    "directories_done",
    "linked_bytes",
    "copied_bytes",
    "elapsed_seconds",
    "capacity_bytes",
    "used_bytes",
    "free_bytes",
)
MONOTONIC_PROGRESS_FIELDS = (
    "entries_done",
    "non_directories_done",
    "directories_done",
    "linked_bytes",
    "copied_bytes",
    "elapsed_seconds",
)
PR_SET_PDEATHSIG = 1
PARENT_DEATH_SIGNAL = signal.SIGUSR1
GROUP_REAP_TIMEOUT_SECONDS = 5


class OutputUnavailable(RuntimeError):
    """The only output sink failed, so supervision must fail closed."""


class Reporter:
    def __init__(self, log_file: Path | None) -> None:
        self._log: TextIO | None = None
        self._stdout_available = True
        if log_file is not None:
            self._log = log_file.open("x", encoding="utf-8")

    def _write_log(self, text: str) -> None:
        if self._log is not None:
            self._log.write(text)
            self._log.flush()

    def write(self, text: str) -> None:
        self._write_log(text)
        if not self._stdout_available:
            return
        try:
            sys.stdout.write(text)
            sys.stdout.flush()
        except (BrokenPipeError, OSError):
            self._stdout_available = False
            self._write_log("MIGRATE_WATCHDOG stage=stdout_unavailable\n")
            if self._log is None:
                raise OutputUnavailable("watchdog stdout is unavailable")

    def emit(self, message: str) -> None:
        self.write(f"MIGRATE_WATCHDOG {message}\n")

    def close(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None


def emit_thread_backtraces(reporter: Reporter, pid: int) -> None:
    reporter.emit(f"stage=backtrace_capture pid={pid}")
    gdb = shutil.which("gdb")
    if gdb:
        result = subprocess.run(
            [gdb, "--batch", "--quiet", "--pid", str(pid), "--ex", "thread apply all bt"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
        reporter.write(
            f"MIGRATE_WATCHDOG_BACKTRACE source=gdb exit_code={result.returncode}\n"
            f"{result.stdout}"
        )
        if not result.stdout.endswith("\n"):
            reporter.write("\n")
        return

    task_dir = Path(f"/proc/{pid}/task")
    try:
        task_ids = sorted(path.name for path in task_dir.iterdir())
    except OSError as error:
        reporter.write(f"MIGRATE_WATCHDOG_BACKTRACE source=proc error={error}\n")
        return

    for task_id in task_ids:
        def read_proc(name: str) -> str:
            try:
                return (task_dir / task_id / name).read_text().strip()
            except OSError as error:
                return f"<unavailable: {error}>"

        reporter.write(
            "MIGRATE_WATCHDOG_BACKTRACE "
            f"source=proc tid={task_id} wchan={read_proc('wchan')!r} "
            f"stack={read_proc('stack')!r}\n"
        )


def process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def signal_process_group(pgid: int, signum: signal.Signals) -> None:
    try:
        os.killpg(pgid, signum)
    except ProcessLookupError:
        pass


def reap_process_group(process: subprocess.Popen[object]) -> None:
    pgid = process.pid
    if process_group_exists(pgid):
        signal_process_group(pgid, signal.SIGKILL)

    process.wait(timeout=GROUP_REAP_TIMEOUT_SECONDS)
    deadline = time.monotonic() + GROUP_REAP_TIMEOUT_SECONDS
    while process_group_exists(pgid) and time.monotonic() < deadline:
        time.sleep(0.01)
    if process_group_exists(pgid):
        raise RuntimeError(f"migration process group {pgid} survived reaping")


def arm_parent_death_signal() -> None:
    parent_pid = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_PDEATHSIG, int(PARENT_DEATH_SIGNAL)) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)


def guardian_parent_died(_signum: int, _frame: object) -> None:
    os.killpg(0, signal.SIGKILL)


def run_guardian(command: list[str]) -> int:
    signal.signal(PARENT_DEATH_SIGNAL, guardian_parent_died)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    child = subprocess.Popen(command)
    return child.wait()


def guardian_command(command: list[str]) -> list[str]:
    return [sys.executable, os.fspath(Path(__file__).resolve()), "--_guardian", "--", *command]


@dataclass
class ProgressMonitor:
    reporter: Reporter
    last_advanced_at: float
    last_values: dict[str, int] | None = None
    last_work: tuple[int, int, int, int] | None = None
    stall_reported: bool = False

    def observe(self, line: str, now: float) -> None:
        if not line.startswith("MIGRATE_PROGRESS "):
            return

        fields = {}
        for item in line.removeprefix("MIGRATE_PROGRESS ").split():
            name, separator, value = item.partition("=")
            if not separator:
                self.reporter.emit("stage=progress_invalid reason=malformed_field")
                return
            fields[name] = value

        if set(PROGRESS_FIELDS) - fields.keys():
            self.reporter.emit("stage=progress_invalid reason=missing_field")
            return

        try:
            values = {name: int(fields[name]) for name in PROGRESS_FIELDS}
        except ValueError:
            self.reporter.emit("stage=progress_invalid reason=non_numeric")
            return

        if (
            any(value < 0 for value in values.values())
            or values["entries_total"] == 0
            or values["entries_done"] > values["entries_total"]
            or values["non_directories_done"] + values["directories_done"]
            > values["entries_done"]
            or values["capacity_bytes"] == 0
            or values["used_bytes"] + values["free_bytes"] != values["capacity_bytes"]
        ):
            self.reporter.emit("stage=progress_invalid reason=invalid_accounting")
            return

        if self.last_values is not None:
            if values["entries_total"] != self.last_values["entries_total"]:
                self.reporter.emit("stage=progress_invalid reason=entries_total_changed")
                return
            if any(
                values[name] < self.last_values[name]
                for name in MONOTONIC_PROGRESS_FIELDS
            ):
                self.reporter.emit("stage=progress_invalid reason=counter_regressed")
                return

        work = (
            values["entries_done"],
            values["non_directories_done"],
            values["directories_done"],
            values["linked_bytes"] + values["copied_bytes"],
        )
        if self.last_work is None or any(
            current > previous
            for previous, current in zip(self.last_work, work)
        ):
            self.last_advanced_at = now
            self.stall_reported = False

        self.last_values = values
        self.last_work = work

    def capture_stall_if_needed(
        self, process: subprocess.Popen[object], now: float, timeout: float
    ) -> None:
        if not self.stall_reported and now - self.last_advanced_at >= timeout:
            self.reporter.emit(
                f"stage=stall elapsed_seconds={int(now - self.last_advanced_at)}"
            )
            emit_thread_backtraces(self.reporter, process.pid)
            self.stall_reported = True


def run(args: argparse.Namespace) -> int:
    reporter = Reporter(args.log_file)
    process: subprocess.Popen[object] | None = None
    selector: selectors.BaseSelector | None = None
    previous_handlers: dict[int, signal.Handlers] = {}
    try:
        started = time.monotonic()
        process = subprocess.Popen(
            guardian_command(args.command),
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=arm_parent_death_signal,
        )
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        pending_output = ""
        progress = ProgressMonitor(reporter, started)
        cancel_reason: str | None = None
        cancel_deadline: float | None = None
        term_deadline: float | None = None

        def forward_child_output() -> None:
            nonlocal pending_output
            assert selector is not None
            for key, _ in selector.select(timeout=0):
                data = os.read(key.fileobj.fileno(), 64 * 1024)
                if not data:
                    selector.unregister(key.fileobj)
                    continue
                text = data.decode(errors="replace")
                reporter.write(text)
                pending_output += text
                while "\n" in pending_output:
                    line, pending_output = pending_output.split("\n", 1)
                    progress.observe(line, time.monotonic())

        def start_graceful_cancel(reason: str) -> None:
            nonlocal cancel_reason, cancel_deadline
            if cancel_reason is not None:
                return
            cancel_reason = reason
            cancel_deadline = time.monotonic() + args.cancel_grace_seconds
            reporter.emit(f"stage=graceful_cancel reason={reason} signal=SIGINT")
            signal_process_group(process.pid, signal.SIGINT)

        def request_cancel(signum: int, _frame: object) -> None:
            reporter.emit(f"stage=cancel_requested signal={signal.Signals(signum).name}")
            start_graceful_cancel("external")

        previous_handlers = {
            signum: signal.signal(signum, request_cancel)
            for signum in (signal.SIGINT, signal.SIGTERM)
        }
        while process.poll() is None:
            forward_child_output()
            now = time.monotonic()

            if cancel_reason is None:
                progress.capture_stall_if_needed(
                    process, now, args.stall_timeout_seconds
                )
                if now - started >= args.hard_timeout_seconds:
                    reporter.emit(
                        f"stage=hard_timeout elapsed_seconds={int(now - started)}"
                    )
                    start_graceful_cancel("hard_timeout")
            elif term_deadline is None and now >= cancel_deadline:
                emit_thread_backtraces(reporter, process.pid)
                reporter.emit("stage=sigterm")
                signal_process_group(process.pid, signal.SIGTERM)
                term_deadline = time.monotonic() + args.term_grace_seconds
            elif term_deadline is not None and now >= term_deadline:
                reporter.emit("stage=sigkill")
                signal_process_group(process.pid, signal.SIGKILL)
                reap_process_group(process)
                forward_child_output()
                reporter.emit("result=aborted forced=true")
                return 137

            selector.select(timeout=0.05)

        forward_child_output()
        returncode = process.wait()
        if cancel_reason is not None:
            reporter.emit("result=aborted forced=false")
            return 124
        if returncode == 0:
            reporter.emit("result=success")
            return 0
        reporter.emit(f"result=failure child_exit_code={returncode}")
        return returncode if returncode > 0 else 128 - returncode
    except BaseException as error:
        try:
            reporter.emit(f"stage=supervisor_exception type={type(error).__name__}")
        except BaseException:
            pass
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if selector is not None:
            selector.close()
        if process is not None:
            if process.stdout is not None:
                process.stdout.close()
            reap_process_group(process)
        reporter.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bcachefs migrate with progress-aware shutdown supervision."
    )
    parser.add_argument("--hard-timeout-seconds", type=float, default=12 * 60 * 60)
    parser.add_argument("--stall-timeout-seconds", type=float, default=30 * 60)
    parser.add_argument("--cancel-grace-seconds", type=float, default=120)
    parser.add_argument("--term-grace-seconds", type=float, default=30)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--_guardian", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if any(
        timeout <= 0
        for timeout in (
            args.hard_timeout_seconds,
            args.stall_timeout_seconds,
            args.cancel_grace_seconds,
            args.term_grace_seconds,
        )
    ):
        parser.error("timeout values must be positive")
    if not args.command:
        parser.error("command is required after --")
    if args.command[0] == "--":
        args.command.pop(0)
    if not args.command:
        parser.error("command is required after --")
    return args


if __name__ == "__main__":
    try:
        args = parse_args()
        if args._guardian:
            raise SystemExit(run_guardian(args.command))
        raise SystemExit(run(args))
    except KeyboardInterrupt:
        raise SystemExit(124)
