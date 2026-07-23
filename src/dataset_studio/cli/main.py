"""CLI principal do Dataset Studio."""

from __future__ import annotations

import argparse
import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dataset_studio.application import (
    TrainingParams,
    begin_training_record,
    finalize_training_record,
    registry_status,
    resolve_model_reference,
    training_recipe,
    source_status,
    version_status,
)
from dataset_studio.domain import (
    Workspace,
    accept_native_export,
    build_import_tasks,
    build_version,
    create_source,
    create_version,
    list_sources,
    list_versions,
)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Configura o parser de argumentos de linha de comando para o utilitário CLI."""

    parser = argparse.ArgumentParser(
        prog="dataset-studio",
        description="Ferramenta autônoma para organização, revisão e materialização de datasets de visão computacional.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("."),
        help="Caminho raiz do workspace de dados (padrão: diretório atual).",
    )

    subparsers = parser.add_subparsers(dest="command", help="Comandos disponíveis")

    # Source commands (Origens)
    source_parser = subparsers.add_parser("source", aliases=["campaign"], help="Gerenciar Origens de dados")
    source_sub = source_parser.add_subparsers(dest="subcommand")

    sc = source_sub.add_parser("create", help="Criar nova Origem de dados")
    sc.add_argument("--id", required=True, help="ID único da Origem")
    sc.add_argument("--videos-dir", type=Path, default=Path("videos"), help="Diretório de vídeos")
    sc.add_argument("--pattern", default="*.mp4", help="Padrão glob de vídeos")
    sc.add_argument("--classes", nargs="+", default=["objeto"], help="Lista de classes")

    source_sub.add_parser("list", help="Listar Origens de dados")

    ss = source_sub.add_parser("status", help="Status da Origem")
    ss.add_argument("--id", required=True, help="ID da Origem")

    # import tasks
    imp = subparsers.add_parser("build-import", help="Gerar tarefas para Label Studio")
    imp.add_argument("--source", "--campaign", dest="source", required=True, help="ID da Origem de dados")

    # revision accept
    rev = subparsers.add_parser("accept-revision", help="Aceitar revisão de anotação")
    rev.add_argument("--source", "--campaign", dest="source", required=True, help="ID da Origem de dados")
    rev.add_argument("--export", type=Path, required=True, help="JSON nativo exportado do Label Studio")
    rev.add_argument("--revision-id", default=None, help="ID da revisão")
    rev.add_argument("--allow-pending", action="store_true", help="Permite pendências como revisão parcial")

    # Version commands (Versões)
    version_parser = subparsers.add_parser("version", aliases=["release"], help="Gerenciar Versões do dataset")
    version_sub = version_parser.add_subparsers(dest="subcommand")

    vc = version_sub.add_parser("create", help="Criar nova Versão do dataset")
    vc.add_argument("--id", required=True, help="ID único da Versão")
    vc.add_argument("--sources", "--campaigns", dest="sources", nargs="+", required=True, help="Origens a incluir")
    vc.add_argument("--assignments-json", required=True, help="JSON de atribuição de vídeos aos splits")

    vb = version_sub.add_parser("build", help="Materializar Versão em disco")
    vb.add_argument("--id", required=True, help="ID da Versão")

    version_sub.add_parser("list", help="Listar Versões do dataset")

    vs = version_sub.add_parser("status", help="Status da Versão")
    vs.add_argument("--id", required=True, help="ID da Versão")

    vt = version_sub.add_parser("train", help="Configurar e treinar modelo a partir de uma Versão")
    vt.add_argument("--id", required=True, help="ID da Versão")
    vt.add_argument("--model", default="yolo26n.pt", help="Modelo YOLO de partida (.pt ou .yaml)")
    vt.add_argument("--epochs", type=int, default=50, help="Número de épocas")
    vt.add_argument("--imgsz", type=int, default=640, help="Tamanho da imagem")
    vt.add_argument("--batch", type=int, default=-1, help="Tamanho do batch")
    vt.add_argument("--workers", type=int, default=0, help="Número de workers")
    vt.add_argument("--device", default="auto", help="Dispositivo (auto, cpu, 0, etc.)")
    vt.add_argument("--patience", type=int, default=50, help="Épocas de paciência (early stopping)")
    vt.add_argument("--lr0", type=float, default=0.01, help="Taxa de aprendizado inicial")
    vt.add_argument("--optimizer", default="auto", help="Otimizador")
    vt.add_argument("--dry-run", action="store_true", help="Mostra a receita e o comando de treino sem executar")

    registry_parser = subparsers.add_parser(
        "registry", help="Consultar e validar a proveniência"
    )
    registry_sub = registry_parser.add_subparsers(dest="subcommand")
    registry_sub.add_parser("status", help="Validar modelos, datasets e runs")

    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    """Ponto de entrada de execução da interface de linha de comando (CLI)."""
    parsed = parse_args(args)
    ws = Workspace.from_path(parsed.workspace)

    if parsed.command in {"source", "campaign"}:
        if parsed.subcommand == "create":
            path = create_source(
                ws,
                source_id=parsed.id,
                videos_dir=parsed.videos_dir,
                video_pattern=parsed.pattern,
                annotation={"classes": parsed.classes},
            )
            print(f"[OK] Origem criada em: {path}")
        elif parsed.subcommand == "list":
            sources = list_sources(ws)
            print("Origens de dados disponíveis:")
            for s in sources:
                print(f" - {s}")
        elif parsed.subcommand == "status":
            st = source_status(ws, parsed.id)
            print(json.dumps(st, indent=2, ensure_ascii=False))

    elif parsed.command == "build-import":
        output = build_import_tasks(ws, parsed.source)
        print(f"[OK] Import tasks gerado em: {output}")

    elif parsed.command == "accept-revision":
        accepted, report_path = accept_native_export(
            ws,
            parsed.source,
            parsed.export,
            revision_id=parsed.revision_id,
            allow_pending=parsed.allow_pending,
        )
        print(f"[OK] Revisão aceita em: {accepted}")

    elif parsed.command in {"version", "release"}:
        if parsed.subcommand == "create":
            assignments = json.loads(parsed.assignments_json)
            path = create_version(
                ws,
                version_id=parsed.id,
                source_ids=parsed.sources,
                assignments=assignments,
            )
            print(f"[OK] Versão configurada em: {path}")
        elif parsed.subcommand == "build":
            manifest = build_version(ws, parsed.id)
            print(f"[OK] Versão materializada com manifesto: {manifest}")
        elif parsed.subcommand == "list":
            versions = list_versions(ws)
            print("Versões do dataset disponíveis:")
            for v in versions:
                print(f" - {v}")
        elif parsed.subcommand == "status":
            st = version_status(ws, parsed.id)
            print(json.dumps(st, indent=2, ensure_ascii=False))
        elif parsed.subcommand == "train":
            training_id = (
                f"t_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_"
                f"{uuid.uuid4().hex[:6]}"
            )
            params = TrainingParams(
                model=resolve_model_reference(ws, parsed.model),
                epochs=parsed.epochs,
                imgsz=parsed.imgsz,
                batch=parsed.batch,
                workers=parsed.workers,
                device=parsed.device,
                patience=parsed.patience,
                lr0=parsed.lr0,
                optimizer=parsed.optimizer,
                project=str(ws.root / "runs" / "detect"),
                name=training_id,
            )
            recipe = training_recipe(ws, parsed.id, params)
            print("=" * 60)
            print(" RECEITA E PARÂMETROS DE TREINAMENTO CONFIGURADOS")
            print("=" * 60)
            print(json.dumps(recipe, indent=2, ensure_ascii=False))
            print("=" * 60)
            if parsed.dry_run:
                print("[DRY-RUN] NENHUM PROCESSO DE TREINO FOI INICIADO.")
            else:
                print(f"[INICIANDO TREINAMENTO]: {recipe['command_str']}")
                begin_training_record(ws, training_id, parsed.id, params)
                try:
                    subprocess.run(recipe["command"], check=True)
                except subprocess.CalledProcessError:
                    finalize_training_record(ws, training_id, "failed")
                    raise
                else:
                    finalize_training_record(ws, training_id, "completed")
    elif parsed.command == "registry":
        if parsed.subcommand == "status":
            status = registry_status(ws)
            print(json.dumps(status, indent=2, ensure_ascii=False))
            return 0 if status["valid"] else 1
    else:
        print("Dataset Studio CLI v0.1.0. Use --help para ver os comandos disponíveis.")


if __name__ == "__main__":
    main()
