"""Supervisão e encerramento seguro de processos e subprocessos."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from typing import Any


def process_group_options(*, hidden: bool = True) -> dict[str, Any]:
    """Retorna flags de criação de subprocesso adequadas para isolamento de grupo por SO."""

    options: dict[str, Any] = {}
    if sys.platform == "win32":
        creationflags = 0
        if hidden:
            creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
        creationflags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        options["creationflags"] = creationflags
    else:
        options["start_new_session"] = True
    return options


def terminate_process_tree(process: subprocess.Popen | None) -> None:
    """Encerra com segurança um processo e toda a sua árvore de subprocessos filhos."""

    if process is None or process.poll() is not None:
        return
    pid = process.pid
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception:
            pass
    else:
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
        except Exception:
            try:
                process.terminate()
            except Exception:
                pass
    try:
        process.wait(timeout=5)
    except Exception:
        pass
