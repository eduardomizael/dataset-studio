"""Supervisão e inicialização local do Label Studio e seu ML backend."""

from __future__ import annotations

import os
import socket
from pathlib import Path


def get_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "localhost"
    finally:
        s.close()
    return ip


def build_label_studio_env(
    local_files_root: str | Path,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    document_root = str(Path(local_files_root).resolve())

    env["LOCAL_FILES_SERVING_ENABLED"] = "true"
    env["LOCAL_FILES_DOCUMENT_ROOT"] = document_root
    env["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"
    env["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = document_root
    return env
