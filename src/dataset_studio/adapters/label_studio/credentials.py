"""Persistência local da credencial única de integração com o Label Studio."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dataset_studio.domain.errors import WorkflowError

API_KEY_ENV = "DATASET_STUDIO_LABEL_STUDIO_API_KEY"
URL_ENV = "DATASET_STUDIO_LABEL_STUDIO_URL"
CREDENTIALS_DIR_ENV = "DATASET_STUDIO_CREDENTIALS_DIR"


@dataclass(frozen=True)
class LabelStudioCredentials:
    base_url: str
    api_key: str
    source: str


def credentials_path() -> Path:
    override = os.environ.get(CREDENTIALS_DIR_ENV)
    if override:
        root = Path(override)
    elif os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        root = Path(os.environ["LOCALAPPDATA"]) / "DatasetStudio"
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        root = root / "dataset-studio"
    return root / "label_studio_credentials.json"


def normalize_base_url(value: str) -> str:
    url = value.strip().rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise WorkflowError("A URL do Label Studio deve começar com http:// ou https://.")
    return url


def load_label_studio_credentials() -> LabelStudioCredentials | None:
    env_key = os.environ.get(API_KEY_ENV, "").strip()
    if env_key:
        return LabelStudioCredentials(
            base_url=normalize_base_url(
                os.environ.get(URL_ENV, "http://127.0.0.1:8080")
            ),
            api_key=env_key,
            source="environment",
        )

    path = credentials_path()
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"Credencial local do Label Studio inválida: {path}") from exc
    api_key = str(payload.get("api_key") or "").strip()
    if not api_key:
        return None
    return LabelStudioCredentials(
        base_url=normalize_base_url(
            str(payload.get("base_url") or "http://127.0.0.1:8080")
        ),
        api_key=api_key,
        source="local",
    )


def save_label_studio_credentials(
    base_url: str, api_key: str
) -> LabelStudioCredentials:
    key = api_key.strip()
    if len(key) < 5:
        raise WorkflowError("O token do Label Studio deve ter pelo menos 5 caracteres.")
    credentials = LabelStudioCredentials(
        base_url=normalize_base_url(base_url),
        api_key=key,
        source="local",
    )
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {"base_url": credentials.base_url, "api_key": credentials.api_key},
            indent=2,
        ),
        encoding="utf-8",
    )
    try:
        temporary.chmod(0o600)
    except OSError:
        pass
    temporary.replace(path)
    return credentials


def delete_label_studio_credentials() -> None:
    credentials_path().unlink(missing_ok=True)


def public_credentials_status() -> dict[str, object]:
    credentials = load_label_studio_credentials()
    return {
        "configured": credentials is not None,
        "base_url": (
            credentials.base_url if credentials else "http://127.0.0.1:8080"
        ),
        "source": credentials.source if credentials else None,
    }
