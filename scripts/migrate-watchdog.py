#!/usr/bin/env python3
"""Progress-aware supervisor for a bcachefs migration subprocess."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass


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


def emit(message: str) -> None:
    print(f"MIGRATE_WATCHDOG {message}", flush=True)


def emit_thread_backtraces(pid: int) -> None:
    emit(f"stage=backtrace_capture pid={pid}")
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
        print(
            f"MIGRATE_WATCHDOG_BACKTRACE source=gdb exit_code={result.returncode}\n{result.stdout}",
            end="" if result.stdout.endswith("\n") else "\n",
            flush=True,
        )
        return

    task_dir = Path(f"/proc/{pid}/task")
    try:
        task_ids = sorted(path.name for path in task_dir.iterdir())
    except OSError as error:
        print(f"MIGRATE_WATCHDOG_BACKTRACE source=proc error={error}", flush=True)
        return

    for task_id in task_ids:
        def read_proc(name: str) -> str:
            try:
                return (task_dir / task_id / name).read_text().strip()
            except OSError as error:
                return f"<unavailable: {error}>"

        print(
            "MIGRATE_WATCHDOG_BACKTRACE "
            f"source=proc tid={task_id} wchan={read_proc('wchan')!r} "
            f"stack={read_proc('stack')!r}",
            flush=True,
        )


def terminate_process_group(process: subprocess.Popen[object], signum: signal.Signals) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signum)


@dataclass
class ProgressMonitor:
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
                emit("stage=progress_invalid reason=malformed_field")
                return
            fields[name] = value

        if set(PROGRESS_FIELDS) - fields.keys():
            emit("stage=progress_invalid reason=missing_field")
            return

        try:
            values = {name: int(fields[name]) for name in PROGRESS_FIELDS}
        except ValueError:
            emit("stage=progress_invalid reason=non_numeric")
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
            emit("stage=progress_invalid reason=invalid_accounting")
            return

        if self.last_values is not None:
            if values["entries_total"] != self.last_values["entries_total"]:
                emit("stage=progress_invalid reason=entries_total_changed")
                return
            if any(
                values[name] < self.last_values[name]
                for name in MONOTONIC_PROGRESS_FIELDS
            ):
                emit("stage=progress_invalid reason=counter_regressed")
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
            emit(f"stage=stall elapsed_seconds={int(now - self.last_advanced_at)}")
            emit_thread_backtraces(process.pid)
            self.stall_reported = True


def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    process = subprocess.Popen(
        args.command,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    pending_output = ""
    progress = ProgressMonitor(started)
    cancel_reason: str | None = None
    cancel_deadline: float | None = None
    term_deadline: float | None = None

    def forward_child_output() -> None:
        nonlocal pending_output
        for key, _ in selector.select(timeout=0):
            data = os.read(key.fileobj.fileno(), 64 * 1024)
            if not data:
                selector.unregister(key.fileobj)
                continue
            text = data.decode(errors="replace")
            sys.stdout.write(text)
            sys.stdout.flush()
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
        emit(f"stage=graceful_cancel reason={reason} signal=SIGINT")
        terminate_process_group(process, signal.SIGINT)

    def request_cancel(signum: int, _frame: object) -> None:
        emit(f"stage=cancel_requested signal={signal.Signals(signum).name}")
        start_graceful_cancel("external")

    previous_handlers = {
        signum: signal.signal(signum, request_cancel)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        while process.poll() is None:
            forward_child_output()
            now = time.monotonic()

            if cancel_reason is None:
                progress.capture_stall_if_needed(
                    process, now, args.stall_timeout_seconds
                )
                if now - started >= args.hard_timeout_seconds:
                    emit(
                        f"stage=hard_timeout elapsed_seconds={int(now - started)}"
                    )
                    start_graceful_cancel("hard_timeout")
            elif term_deadline is None and now >= cancel_deadline:
                emit_thread_backtraces(process.pid)
                emit("stage=sigterm")
                terminate_process_group(process, signal.SIGTERM)
                term_deadline = now + args.term_grace_seconds
            elif term_deadline is not None and now >= term_deadline:
                emit("stage=sigkill")
                terminate_process_group(process, signal.SIGKILL)
                process.wait()
                forward_child_output()
                emit("result=aborted forced=true")
                return 137

            selector.select(timeout=0.05)

        forward_child_output()
        returncode = process.wait()
        if cancel_reason is not None:
            emit("result=aborted forced=false")
            return 124
        if returncode == 0:
            emit("result=success")
            return 0
        emit(f"result=failure child_exit_code={returncode}")
        return returncode if returncode > 0 else 128 - returncode
    finally:
        selector.close()
        process.stdout.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bcachefs migrate with progress-aware shutdown supervision."
    )
    parser.add_argument("--hard-timeout-seconds", type=float, default=12 * 60 * 60)
    parser.add_argument("--stall-timeout-seconds", type=float, default=30 * 60)
    parser.add_argument("--cancel-grace-seconds", type=float, default=120)
    parser.add_argument("--term-grace-seconds", type=float, default=30)
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
        raise SystemExit(run(parse_args()))
    except KeyboardInterrupt:
        raise SystemExit(124)
