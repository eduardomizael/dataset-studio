from pathlib import Path
from dataset_studio.adapters.label_studio.runner import build_label_studio_env


def test_build_label_studio_env_sets_current_and_legacy_local_file_names(tmp_path: Path):
    env = build_label_studio_env(tmp_path)
    document_root = str(tmp_path.resolve())

    assert env["LOCAL_FILES_SERVING_ENABLED"] == "true"
    assert env["LOCAL_FILES_DOCUMENT_ROOT"] == document_root
    assert env["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] == "true"
    assert env["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] == document_root
