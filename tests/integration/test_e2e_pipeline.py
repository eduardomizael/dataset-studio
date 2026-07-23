import json
from pathlib import Path

from dataset_studio.application import campaign_status, release_status, training_recipe
from dataset_studio.domain import (
    Workspace,
    accept_native_export,
    build_import_tasks,
    build_release,
    create_campaign,
    create_release,
    frame_manifest_path,
)
from dataset_studio.ports.trainer import TrainingParams


def test_e2e_full_dataset_studio_pipeline(tmp_path: Path):
    ws = Workspace.from_path(tmp_path)

    # 1. Vídeos e Campanha
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    (videos_dir / "vid_normal.mp4").write_bytes(b"normal_video_bytes")
    (videos_dir / "vid_val.mp4").write_bytes(b"val_video_bytes")

    camp_config = create_campaign(
        ws,
        campaign_id="piloto_2026",
        videos_dir=videos_dir,
        video_pattern="*.mp4",
        annotation={"classes": ["peixe"]},
    )
    assert camp_config.exists()

    # Simular extração de frames
    images_dir = ws.campaign_root("piloto_2026") / "frames" / "raw" / "images"
    (images_dir / "vid_normal_f000001.jpg").write_bytes(b"frame1_bytes")
    (images_dir / "vid_val_f000001.jpg").write_bytes(b"frame2_bytes")

    manifest = {
        "schema_version": 1,
        "frames": [
            {
                "frame_id": "vid_normal_f000001",
                "image": "vid_normal_f000001.jpg",
                "source_video": "vid_normal.mp4",
                "frame_index": 1,
                "width": 1920,
                "height": 1080,
                "predictions": [],
            },
            {
                "frame_id": "vid_val_f000001",
                "image": "vid_val_f000001.jpg",
                "source_video": "vid_val.mp4",
                "frame_index": 1,
                "width": 1920,
                "height": 1080,
                "predictions": [],
            },
        ],
    }
    frame_manifest_path(ws, "piloto_2026").write_text(json.dumps(manifest), encoding="utf-8")

    # 2. Tarefas de Importação para Label Studio
    tasks_file = build_import_tasks(ws, "piloto_2026")
    assert tasks_file.exists()

    # 3. Anotações e Revisão Aceita
    export_json = tmp_path / "export_ls.json"
    ls_export_data = [
        {
            "data": {"frame_id": "vid_normal_f000001"},
            "annotations": [
                {
                    "was_cancelled": False,
                    "result": [
                        {
                            "type": "rectanglelabels",
                            "value": {
                                "x": 10.0,
                                "y": 20.0,
                                "width": 30.0,
                                "height": 40.0,
                                "rectanglelabels": ["peixe"],
                            },
                        }
                    ],
                }
            ],
        },
        {
            "data": {"frame_id": "vid_val_f000001"},
            "annotations": [{"was_cancelled": False, "result": []}],
        },
    ]
    export_json.write_text(json.dumps(ls_export_data), encoding="utf-8")

    accepted_file, report_file = accept_native_export(
        ws, "piloto_2026", export_json, revision_id="rev_v1"
    )
    assert accepted_file.exists()
    assert report_file.exists()

    st_camp = campaign_status(ws, "piloto_2026")
    assert st_camp["next_action"] == "ready-for-release"
    assert st_camp["latest_annotation_revision"] == "rev_v1"

    # 4. Release e Materialização
    rel_config = create_release(
        ws,
        release_id="rel_piloto_v1",
        campaign_ids=["piloto_2026"],
        assignments={
            "train": ["piloto_2026/vid_normal.mp4"],
            "val": ["piloto_2026/vid_val.mp4"],
        },
        annotation_revisions={"piloto_2026": "rev_v1"},
        evaluation_level="pilot",
    )
    assert rel_config.exists()

    manifest_csv = build_release(ws, "rel_piloto_v1")
    assert manifest_csv.exists()
    assert (manifest_csv.parent / "data.yaml").exists()

    st_rel = release_status(ws, "rel_piloto_v1")
    assert st_rel["materialized"] is True

    # 5. Receita de Treinamento Configurável
    custom_params = TrainingParams(
        model="yolo26n.pt",
        epochs=30,
        imgsz=640,
        batch=8,
        device="cpu",
        patience=10,
        optimizer="Adam",
    )
    recipe = training_recipe(ws, "rel_piloto_v1", custom_params)

    assert recipe["data_yaml"] == str(manifest_csv.parent / "data.yaml")
    assert recipe["params"]["epochs"] == 30
    assert recipe["params"]["optimizer"] == "Adam"
    assert "data=" in recipe["command_str"]
    assert "epochs=30" in recipe["command_str"]
