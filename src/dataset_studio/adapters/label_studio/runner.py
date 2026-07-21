"""Supervisão e inicialização local do Label Studio e seu ML backend."""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any

from dataset_studio.domain import WorkflowError, load_source, load_yaml


def get_local_ip() -> str:
    """Detecta o endereço IP local da máquina na rede."""

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
    """Constrói as variáveis de ambiente para o Label Studio servir arquivos locais com segurança."""

    env = dict(os.environ if base_env is None else base_env)
    document_root = str(Path(local_files_root).resolve())

    env["LOCAL_FILES_SERVING_ENABLED"] = "true"
    env["LOCAL_FILES_DOCUMENT_ROOT"] = document_root
    env["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"
    env["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = document_root
    return env


def is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    """Verifica se uma determinada porta TCP está aberta e respondendo."""

    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except (OSError, ConnectionRefusedError):
        return False


def wait_for_port(port: int, host: str = "127.0.0.1", timeout: float = 10.0) -> bool:
    """Aguardar até que uma porta TCP fique disponível dentro do tempo limite (timeout)."""

    import time
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if is_port_open(port, host=host):
            return True
        time.sleep(0.2)
    return False


def wait_for_ml_backend(port: int = 9090, timeout: float = 10.0) -> bool:
    """Confirma que a porta pertence a um backend saudável do Dataset Studio."""
    import json
    import time
    import urllib.request

    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("status") == "UP" and payload.get("model_version"):
                return True
        except Exception:
            pass
        time.sleep(0.2)
    return False


def find_label_studio_executable() -> str | None:
    """Localiza o executável do Label Studio no ambiente Python atual ou no PATH."""
    import shutil
    import sys

    # 1. Procura no diretório do interpretador Python ativo (ex: .venv/Scripts/ no Windows ou .venv/bin/ no Linux)
    env_dir = Path(sys.executable).parent
    for candidate in ["label-studio.exe", "label-studio", "label-studio.cmd", "label-studio.bat"]:
        target = env_dir / candidate
        if target.is_file():
            return str(target)

    # 2. Busca via shutil.which no PATH do sistema
    for candidate in ["label-studio.exe", "label-studio"]:
        found = shutil.which(candidate)
        if found:
            return found

    return None


def start_label_studio_job(
    job_manager: Any,
    ws: Any,
    campaign_id: str,
    port: int = 8080,
) -> dict[str, Any]:
    """Inicia o processo local do Label Studio em segundo plano via JobManager."""

    # Se a porta já estiver aberta, não precisa reacender o serviço
    if is_port_open(port):
        return {"status": "running", "port": port}

    for job in job_manager.list():
        if job["target"] == "label-studio" and job["status"] == "running":
            return {"status": "running", "port": port, "job_id": job["id"]}

    images_dir = ws.campaign_root(campaign_id) / "frames" / "raw" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    doc_root = getattr(ws, "root", images_dir)
    env = build_label_studio_env(doc_root)

    ls_exec = find_label_studio_executable()
    if ls_exec is None:
        from dataset_studio.domain import WorkflowError
        raise WorkflowError(
            "O executável do Label Studio não foi encontrado no ambiente Python atual. "
            "Certifique-se de instalar o pacote executando 'pip install label-studio' ou 'uv pip install label-studio'."
        )

    cmd = [ls_exec, "start", "--port", str(port), "--no-browser"]

    job = job_manager.start(
        cmd,
        kind="label-studio",
        target="label-studio",
        cwd=ws.campaign_root(campaign_id),
        env=env,
    )
    return job


def start_ml_backend_job(
    job_manager: Any,
    ws: Any,
    campaign_id: str,
    model_name: str | None = None,
    port: int = 9090,
) -> dict[str, Any]:
    """Inicia o servidor ML Backend de inferência para auxílio de anotações."""

    import sys

    if is_port_open(port):
        return {"status": "running", "port": port}

    for job in job_manager.list():
        if job["target"] == "ml-backend" and job["status"] == "running":
            return {"status": "running", "port": port, "job_id": job["id"]}

    source = load_source(ws, campaign_id)
    annotation = source.get("annotation", {})
    configured_model = model_name or annotation.get("model")
    if not configured_model:
        raise WorkflowError("Selecione um modelo para iniciar o backend de predicao.")
    resolved = ws.resolve_path(configured_model).resolve()
    try:
        resolved.relative_to(ws.models_root.resolve())
    except ValueError as exc:
        raise WorkflowError("O modelo do backend deve estar dentro de models/.") from exc
    if not resolved.is_file():
        raise WorkflowError(f"Modelo do backend nao encontrado: {resolved}")

    detection: dict[str, Any] = {}
    roi_points: list[list[int]] = []
    detection_config = annotation.get("detection_config")
    if detection_config:
        config_payload = load_yaml(ws.resolve_path(detection_config))
        detection = dict(config_payload.get("detection") or {})
        roi = dict(config_payload.get("roi") or {})
        if roi.get("enabled") and isinstance(roi.get("points"), list):
            roi_points = roi["points"]

    env = build_label_studio_env(ws.root)
    import json
    env.update(
        {
            "DATASET_STUDIO_ML_MODEL": str(resolved),
            "DATASET_STUDIO_ML_CLASSES": json.dumps(annotation.get("classes") or ["objeto"]),
            "DATASET_STUDIO_ML_ROOT": str(ws.root),
            "DATASET_STUDIO_ML_CONF": str(detection.get("conf_threshold", 0.25)),
            "DATASET_STUDIO_ML_DEVICE": str(detection.get("device") or ""),
            "DATASET_STUDIO_ML_OPTIONS": json.dumps(
                {
                    key: value
                    for key, value in {
                        "imgsz": detection.get("img_size"),
                        "iou": detection.get("iou_threshold"),
                        "max_det": detection.get("max_det"),
                        "half": detection.get("half_precision"),
                    }.items()
                    if value is not None
                }
            ),
            "DATASET_STUDIO_ML_ROI": json.dumps(roi_points),
        }
    )

    script = f"""
import json
import os
import uvicorn
from pathlib import Path
from dataset_studio.adapters.label_studio.ml_backend import create_app, GenericLabelStudioBackend
from dataset_studio.adapters.ultralytics.predictor import UltralyticsPredictor

predictor = UltralyticsPredictor(
    os.environ["DATASET_STUDIO_ML_MODEL"],
    conf=float(os.environ["DATASET_STUDIO_ML_CONF"]),
    device=os.environ.get("DATASET_STUDIO_ML_DEVICE") or None,
    inference_options=json.loads(os.environ["DATASET_STUDIO_ML_OPTIONS"]),
    roi_points=json.loads(os.environ["DATASET_STUDIO_ML_ROI"]),
)
predictor.load()
backend = GenericLabelStudioBackend(
    predictor=predictor,
    class_names=json.loads(os.environ["DATASET_STUDIO_ML_CLASSES"]),
    default_root=Path(os.environ["DATASET_STUDIO_ML_ROOT"]),
)
app = create_app(backend)

uvicorn.run(app, host="127.0.0.1", port={port}, log_level="warning")
"""
    cmd = [sys.executable, "-c", script]
    job = job_manager.start(
        cmd,
        kind="ml-backend",
        target="ml-backend",
        cwd=ws.campaign_root(campaign_id),
        env=env,
    )
    return job
