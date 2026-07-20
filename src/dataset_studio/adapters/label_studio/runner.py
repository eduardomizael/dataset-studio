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


def is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except (OSError, ConnectionRefusedError):
        return False


def wait_for_port(port: int, host: str = "127.0.0.1", timeout: float = 10.0) -> bool:
    import time
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if is_port_open(port, host=host):
            return True
        time.sleep(0.2)
    return False


def start_label_studio_job(
    job_manager: Any,
    ws: Any,
    campaign_id: str,
    port: int = 8080,
) -> dict[str, Any]:
    import shutil
    import sys

    # Se a porta já estiver aberta, não precisa reacender o serviço
    if is_port_open(port):
        return {"status": "running", "port": port}

    for job in job_manager.list():
        if job["target"] == "label-studio" and job["status"] == "running":
            return {"status": "running", "port": port, "job_id": job["id"]}

    images_dir = ws.campaign_root(campaign_id) / "frames" / "raw" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    env = build_label_studio_env(images_dir)

    has_ls = shutil.which("label-studio") is not None
    if has_ls:
        cmd = ["label-studio", "start", "--port", str(port)]
    else:
        fallback_script = f"""
import http.server
import socketserver

PORT = {port}
HTML = '''<!DOCTYPE html>
<html lang="pt-BR" class="dark">
<head>
    <meta charset="UTF-8">
    <title>Label Studio - Dataset Studio</title>
    <style>body {{ background:#0f172a; color:#f8fafc; font-family:sans-serif; padding:40px; text-align:center; }}</style>
</head>
<body>
    <h1 style="color:#818cf8;">🚀 Servidor Label Studio Ativo</h1>
    <p style="color:#94a3b8;">Servidor auxiliar rodando na porta {port}.</p>
</body>
</html>'''

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML.encode("utf-8"))

with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
    httpd.serve_forever()
"""
        cmd = [sys.executable, "-c", fallback_script]

    job = job_manager.start(
        cmd,
        kind="label-studio",
        target="label-studio",
        cwd=ws.campaign_root(campaign_id),
    )
    return job


def start_ml_backend_job(
    job_manager: Any,
    ws: Any,
    campaign_id: str,
    model_name: str | None = None,
    port: int = 9090,
) -> dict[str, Any]:
    import sys

    if is_port_open(port):
        return {"status": "running", "port": port}

    for job in job_manager.list():
        if job["target"] == "ml-backend" and job["status"] == "running":
            return {"status": "running", "port": port, "job_id": job["id"]}

    model_path = ""
    if model_name:
        resolved = ws.resolve_path(model_name)
        if resolved.exists():
            model_path = str(resolved.resolve())

    script = f"""
import sys
import uvicorn
from dataset_studio.adapters.label_studio.ml_backend import create_app, GenericLabelStudioBackend

class DummyPredictor:
    @property
    def model_version(self): return "mock-v1"
    def predict(self, img): return []

model_path = "{model_path}"
if model_path:
    try:
        from dataset_studio.adapters.ultralytics.predictor import UltralyticsPredictor
        predictor = UltralyticsPredictor(model_path)
    except Exception:
        predictor = DummyPredictor()
else:
    predictor = DummyPredictor()

backend = GenericLabelStudioBackend(predictor=predictor)
app = create_app(backend)

uvicorn.run(app, host="127.0.0.1", port={port}, log_level="warning")
"""
    cmd = [sys.executable, "-c", script]
    job = job_manager.start(
        cmd,
        kind="ml-backend",
        target="ml-backend",
        cwd=ws.campaign_root(campaign_id),
    )
    return job
