"""
keep_awake.py  (v27)
====================
While a research run is going, ask the operating system not to fall asleep when the computer is idle. A run cannot
continue while the computer sleeps (the server process is frozen), so without this a long run simply stopped while
its clock kept counting.

    hold()      a run started        (counted: several runs share one request)
    release()   a run finished       (the request is dropped when no run is left)

Windows: SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) from a dedicated thread (no extra package).
macOS:   `caffeinate -i -w <server pid>`.
Linux:   `systemd-inhibit --what=idle:sleep ... sleep infinity` when systemd is present.
It only prevents IDLE sleep: closing the laptop lid or shutting down still stops the run (the page then says the
computer was asleep and does not count that time). Switch off with BIOSYN_KEEP_AWAKE=0.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from typing import Any, Dict, Optional

_LOCK = threading.Lock()
_COUNT = 0
_PROC: Optional[subprocess.Popen] = None
_WIN_STOP: Optional[threading.Event] = None
STATE: Dict[str, Any] = {"active": False, "method": None}


def _enabled() -> bool:
    return os.getenv("BIOSYN_KEEP_AWAKE", "1") != "0" and not os.getenv("BIOSYN_OFFLINE")


def _start() -> None:
    global _PROC, _WIN_STOP
    if sys.platform.startswith("win"):
        import ctypes
        stop = threading.Event()

        def hold_thread() -> None:            # the request belongs to the calling thread: keep one thread alive
            ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
            try:
                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
                stop.wait()
            finally:
                try:
                    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
                except Exception:
                    pass

        threading.Thread(target=hold_thread, name="keep-awake", daemon=True).start()
        _WIN_STOP = stop
        STATE.update(active=True, method="SetThreadExecutionState")
        return
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        _PROC = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
        STATE.update(active=True, method="caffeinate")
        return
    if shutil.which("systemd-inhibit"):
        _PROC = subprocess.Popen(["systemd-inhibit", "--what=idle:sleep", "--who=BioSynthetica",
                                  "--why=research run in progress", "--mode=block", "sleep", "infinity"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        STATE.update(active=True, method="systemd-inhibit")
        return
    STATE.update(active=False, method="not available on this system")


def _stop() -> None:
    global _PROC, _WIN_STOP
    if _WIN_STOP is not None:
        _WIN_STOP.set()
        _WIN_STOP = None
    if _PROC is not None:
        try:
            _PROC.terminate()
        except Exception:
            pass
        _PROC = None
    STATE.update(active=False)


def hold() -> Dict[str, Any]:
    global _COUNT
    if not _enabled():
        return {"active": False, "method": "switched off"}
    with _LOCK:
        _COUNT += 1
        if _COUNT == 1:
            try:
                _start()
            except Exception as exc:
                STATE.update(active=False, method=f"failed: {type(exc).__name__}")
        return dict(STATE)


def release() -> None:
    global _COUNT
    with _LOCK:
        if _COUNT > 0:
            _COUNT -= 1
            if _COUNT == 0:
                _stop()
