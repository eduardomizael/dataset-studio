from pathlib import Path
from dataset_studio.adapters.label_studio.runner import (
    build_label_studio_env,
    find_label_studio_executable,
)


def test_build_label_studio_env_sets_current_and_legacy_local_file_names(tmp_path: Path):
    env = build_label_studio_env(tmp_path)
    document_root = str(tmp_path.resolve())

    assert env["LOCAL_FILES_SERVING_ENABLED"] == "true"
    assert env["LOCAL_FILES_DOCUMENT_ROOT"] == document_root
    assert env["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] == "true"
    assert env["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] == document_root


def test_find_label_studio_executable_returns_valid_path_or_none():
    exe = find_label_studio_executable()
    # No ambiente com label-studio instalado, deve retornar uma string contendo label-studio
    if exe is not None:
        assert "label-studio" in exe.lower()

