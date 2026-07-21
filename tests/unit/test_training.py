from pathlib import Path


from dataset_studio.application import (
    TrainingParams,
    training_recipe,
)
from dataset_studio.cli.main import main
from dataset_studio.domain import (
    Workspace,
    accept_native_export,
    build_release,
    create_campaign,
    create_release,
    frame_manifest_path,
)


def create_materialized_release(tmp_path: Path) -> Workspace:
    ws = Workspace.from_path(tmp_path)
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "video1.mp4").write_bytes(b"vid1")
    (videos / "video2.mp4").write_bytes(b"vid2")

    create_campaign(ws, campaign_id="camp1", videos_dir=videos, video_pattern="*.mp4")
    camp_dir = ws.campaign_root("camp1")
    images = camp_dir / "frames" / "raw" / "images"
    (images / "img1.jpg").write_bytes(b"img1")
    (images / "img2.jpg").write_bytes(b"img2")

    manifest = {
        "schema_version": 1,
        "frames": [
            {
                "frame_id": "f1",
                "image": "img1.jpg",
                "source_video": "video1.mp4",
                "frame_index": 1,
                "width": 640,
                "height": 480,
                "predictions": [],
            },
            {
                "frame_id": "f2",
                "image": "img2.jpg",
                "source_video": "video2.mp4",
                "frame_index": 1,
                "width": 640,
                "height": 480,
                "predictions": [],
            },
        ],
    }
    frame_manifest_path(ws, "camp1").write_text(
        __import__("json").dumps(manifest), encoding="utf-8"
    )

    exported = tmp_path / "export.json"
    tasks = [
        {"data": {"frame_id": "f1"}, "annotations": [{"was_cancelled": False, "result": []}]},
        {"data": {"frame_id": "f2"}, "annotations": [{"was_cancelled": False, "result": []}]},
    ]
    exported.write_text(__import__("json").dumps(tasks), encoding="utf-8")
    accept_native_export(ws, "camp1", exported)

    create_release(
        ws,
        release_id="rel1",
        campaign_ids=["camp1"],
        assignments={"train": ["camp1/video1.mp4"], "val": ["camp1/video2.mp4"]},
    )
    build_release(ws, "rel1")
    return ws


def test_training_params_customization(tmp_path: Path):
    ws = create_materialized_release(tmp_path)
    params = TrainingParams(
        model="custom_model.pt",
        epochs=100,
        imgsz=1280,
        batch=16,
        workers=4,
        device="0",
        patience=20,
        lr0=0.005,
        optimizer="AdamW",
    )
    recipe = training_recipe(ws, "rel1", params)

    assert recipe["release_id"] == "rel1"
    assert recipe["params"]["model"] == "custom_model.pt"
    assert recipe["params"]["epochs"] == 100
    assert recipe["params"]["imgsz"] == 1280
    assert recipe["params"]["batch"] == 16
    assert recipe["params"]["device"] == "0"
    assert recipe["params"]["optimizer"] == "AdamW"
    assert "data=" in recipe["command_str"]
    assert "epochs=100" in recipe["command_str"]
    assert "imgsz=1280" in recipe["command_str"]
    assert "optimizer=AdamW" in recipe["command_str"]


def test_cli_release_train_dry_run(tmp_path: Path, capsys):
    create_materialized_release(tmp_path)

    main([
        "--workspace",
        str(tmp_path),
        "release",
        "train",
        "--id",
        "rel1",
        "--model",
        "yolo26n.pt",
        "--epochs",
        "25",
        "--imgsz",
        "640",
        "--device",
        "cpu",
        "--dry-run",
    ])

    captured = capsys.readouterr()
    assert "RECEITA E PARÂMETROS DE TREINAMENTO CONFIGURADOS" in captured.out
    assert "epochs\": 25" in captured.out
    assert "device\": \"cpu\"" in captured.out
    assert "[DRY-RUN] NENHUM PROCESSO DE TREINO FOI INICIADO." in captured.out
