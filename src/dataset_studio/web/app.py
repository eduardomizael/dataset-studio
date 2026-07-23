"""Aplicação Web Local FastAPI do Dataset Studio."""

from __future__ import annotations

import argparse
import atexit
import json
import importlib.util
import re
import shutil
import tempfile
import uuid
import uvicorn
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from dataset_studio.adapters.label_studio.runner import (
    start_label_studio_job,
    start_ml_backend_job,
    wait_for_port,
    wait_for_ml_backend,
)
from dataset_studio.adapters.label_studio.api_client import LabelStudioClient
from dataset_studio.adapters.label_studio.credentials import (
    delete_label_studio_credentials,
    public_credentials_status,
    save_label_studio_credentials,
)
from dataset_studio.adapters.opencv.media import extract_source_frames, preannotate_source_frames
from dataset_studio.application import (
    JobManager,
    TrainingParams,
    begin_training_record,
    ensure_label_studio_project,
    export_deployment_bundle,
    finalize_training_record,
    inspect_finished_tasks,
    label_studio_integration_status,
    list_available_models,
    preview_split_metrics,
    promote_registered_model,
    registry_status,
    resolve_model_reference,
    source_status,
    training_recipe,
    version_status,
)
from dataset_studio.domain import (
    WorkflowError,
    Workspace,
    accept_native_export,
    build_import_tasks,
    build_version,
    create_source,
    create_version,
    dataset_registry_path,
    delete_source,
    delete_version,
    dump_yaml,
    list_sources,
    list_registered_aliases,
    list_registered_models,
    list_registered_sources,
    list_versions,
    load_source,
    load_yaml,
    register_source_manifest,
    run_registry_path,
    sha256,
    unregister_run,
    validate_id,
)

job_manager = JobManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida do servidor FastAPI e encerramento de subprocessos ao finalizar."""
    yield
    # Ao fechar o servidor do Dataset Studio, encerra os outros servidores (Label Studio e ML Backend)
    job_manager.stop_all(wait=True)



atexit.register(lambda: job_manager.stop_all(wait=False))


class LabelStudioStartReq(BaseModel):
    """Requisição para inicializar o servidor de anotação Label Studio."""

    enable_ml: bool = False
    model: str | None = None
    allow_partial_predictions: bool = False


class LabelStudioSettingsReq(BaseModel):
    """Credencial configurada uma única vez para a instância do Label Studio."""

    base_url: str = "http://127.0.0.1:8080"
    api_key: str = Field(min_length=5, repr=False)


class LabelStudioPrepareReq(BaseModel):
    """Opções explícitas para preparar um projeto já iniciado."""

    allow_partial_predictions: bool = False



class SourceCreateReq(BaseModel):
    """Parâmetros para criação de uma nova fonte de dados (source/campanha)."""

    source_id: str | None = None
    campaign_id: str | None = None
    videos_dir: str = "videos"
    video_pattern: str = "*.mp4"

    video_files: list[str] | None = None
    video_notes: dict[str, str] | None = None
    capture_units: list[dict[str, Any]] | None = None
    classes: list[str] = Field(default_factory=lambda: ["objeto"])

    @property
    def target_id(self) -> str:
        val = self.source_id or self.campaign_id
        if not val:
            raise ValueError("Identificador da origem obrigatório.")
        return val


class ExtractionReq(BaseModel):
    mode: Literal["uniform", "smart"] = "uniform"
    uniform_frame_step: int = Field(default=30, ge=1)
    model: str | None = None
    confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    scan_step: int = Field(default=15, ge=1)
    dense_step: int = Field(default=30, ge=1)
    sparse_step: int = Field(default=90, ge=1)
    margin: int = Field(default=45, ge=0)
    max_negatives_per_video: int = Field(default=15, ge=0)


class ImportTasksReq(BaseModel):
    mode: Literal["existing", "none", "model"] = "existing"
    model: str | None = None
    confidence: float = Field(default=0.25, ge=0.0, le=1.0)

CampaignCreateReq = SourceCreateReq


class ExportAcceptReq(BaseModel):
    """Parâmetros para aceitação e validação de exportação do Label Studio."""

    path: str
    revision_id: str | None = None
    allow_pending: bool = False



class VersionCreateReq(BaseModel):
    """Parâmetros para criação de uma nova versão de dataset (version/release)."""

    version_id: str | None = None
    release_id: str | None = None

    sources: list[str] | None = None
    campaigns: list[str] | None = None
    assignments: dict[str, list[str]]
    annotation_revisions: dict[str, str] = Field(default_factory=dict)
    evaluation_level: Literal["pilot", "standard", "robust"] = "standard"

    @property
    def target_id(self) -> str:
        val = self.version_id or self.release_id
        if not val:
            raise ValueError("Identificador da versão obrigatório.")
        return val

    @property
    def target_sources(self) -> list[str]:
        val = self.sources or self.campaigns
        if not val:
            raise ValueError("Ao menos uma origem deve ser informada.")
        return val


ReleaseCreateReq = VersionCreateReq


class SplitPreviewReq(BaseModel):
    """Parâmetros para pré-visualização de métricas de divisão de dataset (splits)."""

    source_id: str | None = None
    campaign_id: str | None = None
    assignments: dict[str, list[str]]
    revision_id: str | None = None
    evaluation_level: Literal["pilot", "standard", "robust"] = "standard"



class TrainStartReq(BaseModel):
    """Parâmetros para disparar um novo treinamento de modelo YOLO."""

    model: str = "yolo26n.pt"
    epochs: int = 50
    imgsz: int = 640
    batch: int = -1
    workers: int = 0
    device: str = "auto"
    patience: int = 50
    lr0: float = 0.01
    optimizer: str = "auto"


class PromoteModelReq(BaseModel):
    target_name: str | None = None
    overwrite: bool = False


class DeployModelReq(BaseModel):
    deployment_id: str | None = None
    artifact_path: str | None = None



def create_web_app(workspace: Workspace) -> FastAPI:
    """Fábrica para criação e configuração do servidor FastAPI do Dataset Studio Web."""
    app = FastAPI(title="Dataset Studio Web Dashboard", version="0.1.0", lifespan=lifespan)

    def training_ids_for_version(version_id: str) -> list[str]:
        runs_root = workspace.root / "runs" / "detect"
        matches: list[str] = []
        if not runs_root.exists():
            return matches
        for run_dir in runs_root.iterdir():
            job_path = run_dir / "workflow_job.json"
            if not job_path.is_file():
                continue
            try:
                payload = json.loads(job_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (payload.get("metadata") or {}).get("release_id") == version_id:
                matches.append(run_dir.name)
        return sorted(matches)

    def versions_for_source(source_id: str, revision_id: str | None = None) -> list[str]:
        matches: list[str] = []
        for version_id in list_versions(workspace):
            try:
                version = load_yaml(workspace.version_config_path(version_id))
            except WorkflowError:
                continue
            if source_id not in (version.get("sources") or version.get("campaigns") or []):
                continue
            if revision_id is not None and (version.get("annotation_revisions") or {}).get(source_id) != revision_id:
                continue
            matches.append(version_id)
        return matches

    def deletion_impact(resource_type: str, resource_id: str, source_id: str | None = None) -> dict[str, Any]:
        versions: list[str] = []
        trainings: list[str] = []
        if resource_type == "source":
            versions = versions_for_source(resource_id)
        elif resource_type == "revision":
            if not source_id:
                raise WorkflowError("source_id e obrigatorio para revisar o impacto da revisao.")
            versions = versions_for_source(source_id, resource_id)
        elif resource_type == "version":
            versions = [resource_id]
        elif resource_type == "training":
            trainings = [resource_id]
        else:
            raise WorkflowError(f"Tipo de recurso desconhecido: {resource_type}")
        for version_id in versions:
            trainings.extend(training_ids_for_version(version_id))
        shared_video_references: dict[str, list[str]] = {}
        if resource_type == "source":
            source = load_source(workspace, resource_id)
            directory = workspace.resolve_path(source["videos"]["directory"]).resolve()
            target_paths = {
                str((directory / item["name"]).resolve())
                for item in source["videos"].get("files", [])
                if isinstance(item, dict)
            }
            for other_id in list_sources(workspace):
                if other_id == resource_id:
                    continue
                other = load_source(workspace, other_id)
                other_dir = workspace.resolve_path(other["videos"]["directory"]).resolve()
                for item in other["videos"].get("files", []):
                    if not isinstance(item, dict):
                        continue
                    path = str((other_dir / item["name"]).resolve())
                    if path in target_paths:
                        shared_video_references.setdefault(path, []).append(other_id)
        return {
            "resource_type": resource_type,
            "resource_id": resource_id,
            "source_id": source_id,
            "dependent_versions": sorted(set(versions if resource_type in {"source", "revision"} else [])),
            "dependent_trainings": sorted(set(trainings if resource_type != "training" else [])),
            "shared_video_references": shared_video_references,
            "warning": "A exclusao e permanente e pode invalidar a rastreabilidade de recursos mantidos sem cascata.",
        }

    def remove_training(training_id: str) -> None:
        validate_id(training_id, "training_id")
        run_dir = workspace.root / "runs" / "detect" / training_id
        if not run_dir.is_dir():
            raise WorkflowError(f"Treinamento nao encontrado: {training_id}")
        shutil.rmtree(run_dir, ignore_errors=False)
        unregister_run(workspace, training_id)

    def remove_version_with_dependents(version_id: str, cascade: bool) -> None:
        if cascade:
            for training_id in training_ids_for_version(version_id):
                remove_training(training_id)
        delete_version(workspace, version_id)


    @app.get("/", response_class=HTMLResponse)
    def index():
        return """<!DOCTYPE html>
<html lang="pt-BR" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Studio - Início</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background-color: #0f172a; color: #f8fafc; font-family: system-ui, -apple-system, sans-serif; }
    </style>
</head>
<body class="p-6 md:p-10">
    <div class="max-w-7xl mx-auto">
        <!-- Header -->
        <header class="flex flex-col md:flex-row justify-between items-start md:items-center pb-6 mb-8 border-b border-slate-800 gap-4">
            <div>
                <h1 class="text-3xl font-extrabold tracking-tight text-indigo-400">Dataset Studio</h1>
                <p class="text-slate-400 text-sm mt-1">Gerenciamento autônomo de Origens de dados, Versões e Treinamentos YOLO</p>
            </div>
            <div class="flex items-center gap-3">
                <button onclick="openCreateCampaignModal()" class="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-sm rounded-lg shadow-lg shadow-indigo-600/30 transition">
                    + Nova Origem de Dados
                </button>
                <span class="px-3 py-1.5 bg-slate-800 text-slate-400 border border-slate-700 rounded-lg text-xs font-mono">
                    Workspace: <span id="ws-root">...</span>
                </span>
            </div>
        </header>

        <!-- Layout 3 Colunas -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <!-- Coluna 1: Origens -->
            <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col gap-4">
                <div class="flex justify-between items-center pb-2 border-b border-slate-800">
                    <h2 class="text-xl font-bold text-slate-200 flex items-center gap-2">
                        <span>📁</span> Origens de Dados
                    </h2>
                    <button onclick="loadData()" class="text-xs text-indigo-400 hover:underline">🔄 Atualizar</button>
                </div>
                <div id="campaigns-list" class="space-y-4">
                    <p class="text-slate-500 text-sm">Carregando origens...</p>
                </div>
            </div>

            <!-- Coluna 2: Versões -->
            <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col gap-4">
                <div class="flex justify-between items-center pb-2 border-b border-slate-800">
                    <h2 class="text-xl font-bold text-slate-200 flex items-center gap-2">
                        <span>📦</span> Versões do Dataset
                    </h2>
                    <button onclick="loadData()" class="text-xs text-indigo-400 hover:underline">🔄 Atualizar</button>
                </div>
                <div id="releases-list" class="space-y-4">
                    <p class="text-slate-500 text-sm">Carregando versões...</p>
                </div>
            </div>

            <!-- Coluna 3: Treinamentos -->
            <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col gap-4">
                <div class="flex justify-between items-center pb-2 border-b border-slate-800">
                    <h2 class="text-xl font-bold text-slate-200 flex items-center gap-2">
                        <span>⚡</span> Treinamentos
                    </h2>
                    <button onclick="loadData()" class="text-xs text-indigo-400 hover:underline">🔄 Atualizar</button>
                </div>
                <div id="trainings-list" class="space-y-4">
                    <p class="text-slate-500 text-sm">Carregando treinamentos...</p>
                </div>
            </div>
        </div>
    </div>

    <!-- Modal Nova Origem -->
    <div id="modal-campaign" class="fixed inset-0 bg-slate-950/80 backdrop-blur-sm hidden flex items-center justify-center p-4 z-50">
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 max-w-lg w-full shadow-2xl space-y-5">
            <div class="flex justify-between items-center pb-3 border-b border-slate-800">
                <h3 class="text-lg font-bold text-indigo-400">Criar Nova Origem de Dados</h3>
                <button onclick="closeCreateCampaignModal()" class="text-slate-400 hover:text-white">✕</button>
            </div>
            
            <div class="space-y-4 text-sm">
                <div>
                    <label class="block text-slate-300 font-medium mb-1">ID da Origem</label>
                    <input type="text" id="input-campaign-id" placeholder="ex: origem_01" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2.5 text-white focus:border-indigo-500 focus:outline-none">
                </div>
                
                <div>
                    <label class="block text-slate-300 font-medium mb-1.5">Seleção de Vídeos</label>
                    <div class="relative border-2 border-dashed border-slate-700 hover:border-indigo-500 rounded-xl p-5 text-center bg-slate-800/40 transition cursor-pointer" onclick="document.getElementById('input-video-files').click()">
                        <input type="file" id="input-video-files" multiple accept="video/*,.mp4,.avi,.mkv" class="hidden" onchange="handleVideoSelection(this)">
                        <div class="text-3xl mb-1">🎬</div>
                        <div class="text-sm font-medium text-slate-200">Clique aqui para selecionar os vídeos</div>
                        <div class="text-xs text-slate-400 mt-1">Aceita formatos .mp4, .avi, .mkv (vários arquivos)</div>
                        <div id="selected-video-count" class="text-xs text-indigo-400 font-semibold mt-2 hidden"></div>
                    </div>
                </div>

                <!-- Container de Notas Individuais por Vídeo -->
                <div id="video-notes-container" class="hidden space-y-2 max-h-72 overflow-y-auto pr-1">
                    <label class="block text-slate-300 font-medium text-xs">Vídeos e unidades experimentais:</label>
                    <p class="text-[11px] text-slate-400">
                        Um vídeo contínuo pode ser dividido em levas independentes sem recodificação.
                        Use segundos de início e fim; intervalos não utilizados serão excluídos.
                    </p>
                    <div id="video-notes-list" class="space-y-2"></div>
                </div>

                <div>
                    <label class="block text-slate-300 font-medium mb-1">Classes de Objetos (separadas por vírgula)</label>
                    <input type="text" id="input-classes" value="peixe" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2.5 text-white focus:border-indigo-500 focus:outline-none">
                </div>
            </div>

            <div id="modal-error" class="text-rose-400 text-xs hidden"></div>

            <div class="flex justify-end gap-3 pt-3 border-t border-slate-800">
                <button onclick="closeCreateCampaignModal()" class="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg font-medium text-xs">Cancelar</button>
                <button onclick="submitCreateCampaign()" class="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg font-medium text-xs shadow-lg shadow-indigo-600/30">Criar Origem de Dados</button>
            </div>
        </div>
    </div>

    <script>
        function openCreateCampaignModal() {
            document.getElementById('modal-campaign').classList.remove('hidden');
            document.getElementById('modal-error').classList.add('hidden');
        }

        function closeCreateCampaignModal() {
            document.getElementById('modal-campaign').classList.add('hidden');
        }

        function handleVideoSelection(input) {
            const files = input.files;
            const infoDiv = document.getElementById('selected-video-count');
            const notesContainer = document.getElementById('video-notes-container');
            const notesList = document.getElementById('video-notes-list');

            if (files && files.length > 0) {
                const names = Array.from(files).map(f => f.name).join(', ');
                infoDiv.innerText = `✓ ${files.length} vídeo(s) selecionado(s)`;
                infoDiv.classList.remove('hidden');

                let notesHtml = '';
                Array.from(files).forEach((f, idx) => {
                    notesHtml += `
                        <div data-video-card data-video-name="${escapeHtml(f.name)}" data-video-index="${idx}" class="p-2.5 bg-slate-800/60 border border-slate-700/60 rounded-lg space-y-2">
                            <div class="flex justify-between gap-2">
                                <div class="text-xs font-mono font-medium text-indigo-300">${escapeHtml(f.name)}</div>
                                <span data-duration-label class="text-[10px] text-slate-500">Lendo duração...</span>
                            </div>
                            <input type="text" data-video-note-name="${escapeHtml(f.name)}" placeholder="Ex: Iluminação baixa, peixes rápidos, câmera 2..." class="w-full bg-slate-900 border border-slate-700 rounded p-1.5 text-xs text-white focus:border-indigo-500 focus:outline-none">
                            <div class="flex items-center justify-between">
                                <span class="text-[11px] text-slate-400">Sem divisão: o vídeo inteiro será uma unidade.</span>
                                <button type="button" onclick="addCaptureSegment(this)" class="px-2 py-1 bg-indigo-600/20 text-indigo-300 border border-indigo-500/30 rounded text-[10px]">+ Adicionar leva</button>
                            </div>
                            <div data-segments class="space-y-2"></div>
                        </div>
                    `;
                });
                notesList.innerHTML = notesHtml;
                notesContainer.classList.remove('hidden');
                Array.from(files).forEach((file, idx) => {
                    const card = notesList.querySelector(`[data-video-index="${idx}"]`);
                    const preview = document.createElement('video');
                    const objectUrl = URL.createObjectURL(file);
                    preview.preload = 'metadata';
                    preview.onloadedmetadata = () => {
                        const duration = Number(preview.duration || 0);
                        card.setAttribute('data-duration', String(duration));
                        card.querySelector('[data-duration-label]').innerText =
                            duration > 0 ? `Duração: ${duration.toFixed(1)} s` : 'Duração indisponível';
                        URL.revokeObjectURL(objectUrl);
                    };
                    preview.onerror = () => {
                        card.querySelector('[data-duration-label]').innerText = 'Duração indisponível';
                        URL.revokeObjectURL(objectUrl);
                    };
                    preview.src = objectUrl;
                });
            } else {
                infoDiv.classList.add('hidden');
                notesContainer.classList.add('hidden');
                notesList.innerHTML = '';
            }
        }

        function safeUnitId(value) {
            const normalized = value.normalize('NFD').replace(/[\u0300-\u036f]/g, '')
                .replace(/[.][^.]+$/, '').replace(/[^A-Za-z0-9_-]+/g, '_')
                .replace(/^[_-]+|[_-]+$/g, '');
            return normalized || 'unidade';
        }

        function addCaptureSegment(button) {
            const card = button.closest('[data-video-card]');
            const container = card.querySelector('[data-segments]');
            const videoName = card.getAttribute('data-video-name');
            const videoIndex = Number(card.getAttribute('data-video-index') || 0);
            const duration = Number(card.getAttribute('data-duration') || 0);
            const segmentIndex = container.querySelectorAll('.capture-segment-row').length + 1;
            const unitId = `${safeUnitId(videoName)}_leva_${String(segmentIndex).padStart(2, '0')}_${videoIndex + 1}`;
            const row = document.createElement('div');
            row.className = 'capture-segment-row grid grid-cols-1 sm:grid-cols-5 gap-1.5 p-2 bg-slate-950/70 border border-slate-700 rounded';
            row.innerHTML = `
                <input data-unit-id value="${unitId}" placeholder="ID da leva" class="sm:col-span-2 bg-slate-900 border border-slate-700 rounded p-1.5 text-[11px] text-white font-mono">
                <input data-start type="number" min="0" step="0.1" value="0" placeholder="Início (s)" class="bg-slate-900 border border-slate-700 rounded p-1.5 text-[11px] text-white">
                <input data-end type="number" min="0" step="0.1" value="${duration > 0 ? duration.toFixed(1) : ''}" placeholder="Fim (s)" class="bg-slate-900 border border-slate-700 rounded p-1.5 text-[11px] text-white">
                <button type="button" onclick="this.closest('.capture-segment-row').remove()" class="text-rose-400 text-[11px]">Remover</button>
                <input data-unit-note placeholder="Descrição da leva / condição" class="sm:col-span-5 bg-slate-900 border border-slate-700 rounded p-1.5 text-[11px] text-white">
            `;
            container.appendChild(row);
        }

        async function submitCreateCampaign() {
            const id = document.getElementById('input-campaign-id').value.trim();
            const filesInput = document.getElementById('input-video-files');
            const classesRaw = document.getElementById('input-classes').value.trim();
            const errDiv = document.getElementById('modal-error');

            if (!id) {
                errDiv.innerText = 'Preencha o ID da origem.';
                errDiv.classList.remove('hidden');
                return;
            }
            if (!filesInput.files || filesInput.files.length === 0) {
                errDiv.innerText = 'Selecione pelo menos um arquivo de vídeo.';
                errDiv.classList.remove('hidden');
                return;
            }

            const classes = classesRaw ? classesRaw.split(',').map(c => c.trim()).filter(Boolean) : ['objeto'];
            const videoNotes = {};
            document.querySelectorAll('[data-video-note-name]').forEach(inp => {
                const name = inp.getAttribute('data-video-note-name');
                const note = inp.value.trim();
                if (name && note) {
                    videoNotes[name] = note;
                }
            });
            const cards = Array.from(document.querySelectorAll('[data-video-card]'));
            const hasSegments = cards.some(
                card => card.querySelectorAll('.capture-segment-row').length > 0
            );
            const captureUnits = [];
            if (hasSegments) {
                cards.forEach((card, videoIndex) => {
                    const videoName = card.getAttribute('data-video-name');
                    const rows = Array.from(card.querySelectorAll('.capture-segment-row'));
                    if (rows.length === 0) {
                        captureUnits.push({
                            unit_id: `${safeUnitId(videoName)}_completo_${videoIndex + 1}`,
                            source_video: videoName,
                            start_seconds: 0,
                            end_seconds: null,
                            note: videoNotes[videoName] || ''
                        });
                        return;
                    }
                    rows.forEach(row => {
                        const endRaw = row.querySelector('[data-end]').value.trim();
                        captureUnits.push({
                            unit_id: row.querySelector('[data-unit-id]').value.trim(),
                            source_video: videoName,
                            start_seconds: Number(row.querySelector('[data-start]').value),
                            end_seconds: endRaw === '' ? null : Number(endRaw),
                            note: row.querySelector('[data-unit-note]').value.trim()
                        });
                    });
                });
            }

            const formData = new FormData();
            formData.append('source_id', id);
            formData.append('campaign_id', id);
            formData.append('classes', JSON.stringify(classes));
            formData.append('video_notes', JSON.stringify(videoNotes));
            formData.append('capture_units', JSON.stringify(captureUnits));
            for (const file of filesInput.files) {
                formData.append('videos', file);
            }

            try {
                errDiv.innerText = 'Enviando vídeos e criando origem...';
                errDiv.classList.remove('hidden');
                errDiv.classList.remove('text-rose-400');
                errDiv.classList.add('text-indigo-400');

                const res = await fetch('/api/sources/upload', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                if (!res.ok) {
                    throw new Error(data.detail || 'Erro ao criar origem de dados.');
                }
                closeCreateCampaignModal();
                loadData();
            } catch (err) {
                errDiv.innerText = err.message;
                errDiv.classList.remove('hidden');
                errDiv.classList.remove('text-indigo-400');
                errDiv.classList.add('text-rose-400');
            }
        }

        async function loadData() {
            try {
                const resW = await fetch('/api/workspace');
                const ws = await resW.json();
                document.getElementById('ws-root').innerText = ws.root;

                // Origens
                const resC = await fetch('/api/sources');
                const sources = await resC.json();
                const divC = document.getElementById('campaigns-list');
                
                if (sources.length === 0) {
                    divC.innerHTML = `
                        <div class="text-center py-8 border border-dashed border-slate-800 rounded-xl">
                            <p class="text-slate-400 text-sm font-medium">Nenhuma origem criada ainda.</p>
                            <button onclick="openCreateCampaignModal()" class="mt-3 text-xs text-indigo-400 font-semibold hover:underline">
                                + Clique aqui para criar a primeira origem
                            </button>
                        </div>
                    `;
                } else {
                    let html = '';
                    for (const sId of sources) {
                        const resSt = await fetch(`/api/sources/${sId}`);
                        const st = await resSt.json();
                        const targetId = st.source_id || st.campaign_id;
                        
                        html += `
                            <div class="p-4 bg-slate-800/40 border border-slate-800 rounded-xl space-y-2 hover:border-indigo-500/50 hover:bg-slate-800/70 transition group relative">
                                <div class="flex flex-col gap-2">
                                    <div class="flex items-start justify-between gap-2">
                                        <h3 class="font-bold text-indigo-300 text-base group-hover:text-indigo-200 truncate min-w-0 flex-1" title="${escapeHtml(targetId)}">${escapeHtml(targetId)}</h3>
                                        <div class="flex items-center gap-1.5">
                                            <span class="px-2 py-0.5 bg-slate-800 text-indigo-400 border border-indigo-500/30 rounded text-[11px] font-semibold whitespace-nowrap">
                                                ${st.next_action}
                                            </span>
                                            <button onclick="deleteSource(event, '${escapeHtml(targetId)}')" title="Excluir origem do disco" class="p-1 text-slate-500 hover:text-rose-400 transition">🗑️</button>
                                        </div>
                                    </div>
                                    <div class="text-xs text-slate-400">
                                        Vídeos: <span class="text-slate-200 font-mono font-medium">${st.videos}</span> | 
                                        Frames: <span class="text-slate-200 font-mono font-medium">${st.frames}</span> | 
                                        Tasks: <span class="text-slate-200 font-mono font-medium">${st.import_tasks}</span>
                                    </div>
                                </div>
                                <a href="/source.html?id=${targetId}" class="text-xs text-indigo-400 font-medium pt-1 flex items-center gap-1 group-hover:translate-x-1 transition-transform">
                                    <span>Ver detalhes e etapas</span> &rarr;
                                </a>
                            </div>
                        `;
                    }
                    divC.innerHTML = html;
                }

                // Versões
                const resR = await fetch('/api/versions');
                const versions = await resR.json();
                const divR = document.getElementById('releases-list');

                if (versions.length === 0) {
                    divR.innerHTML = '<p class="text-slate-500 text-sm py-4 text-center">Nenhuma versão materializada ainda.</p>';
                } else {
                    divR.innerHTML = versions.map(v => `
                        <div class="p-4 bg-slate-800/40 border border-slate-800 rounded-xl space-y-2 hover:border-emerald-500/50 hover:bg-slate-800/70 transition group">
                            <div class="flex items-start justify-between gap-2">
                                <div class="min-w-0 flex-1 space-y-1">
                                    <div class="font-bold text-emerald-400 group-hover:text-emerald-300 text-sm truncate" title="${escapeHtml(v)}">${escapeHtml(v)}</div>
                                    <div class="text-xs text-slate-400">Clique para ver detalhes, splits e treinar</div>
                                </div>
                                <div class="flex items-center gap-1.5">
                                    <span class="px-2 py-0.5 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded text-[11px] font-semibold whitespace-nowrap">Materializada</span>
                                    <button onclick="deleteRelease(event, '${escapeHtml(v)}')" title="Excluir release e arquivos do disco" class="p-1 text-slate-500 hover:text-rose-400 transition">🗑️</button>
                                </div>
                            </div>
                            <a href="/version.html?id=${v}" class="text-xs text-emerald-400 font-medium pt-1 flex items-center gap-1 group-hover:translate-x-1 transition-transform">
                                <span>Acessar release</span> &rarr;
                            </a>
                        </div>
                    `).join('');
                }

                // Treinamentos
                const resT = await fetch('/api/trainings');
                const trainings = await resT.json();
                const divT = document.getElementById('trainings-list');

                if (!trainings || trainings.length === 0) {
                    divT.innerHTML = '<p class="text-slate-500 text-sm py-4 text-center">Nenhum treinamento realizado ainda.</p>';
                } else {
                    divT.innerHTML = trainings.map(t => `
                        <div class="p-4 bg-slate-800/40 border border-slate-800 rounded-xl space-y-2 hover:border-amber-500/50 hover:bg-slate-800/70 transition group">
                            <div class="flex items-start justify-between gap-2">
                                <div class="min-w-0 flex-1 space-y-1">
                                    <div class="font-bold text-amber-400 group-hover:text-amber-300 text-sm truncate font-mono" title="${escapeHtml(t.name)}">${escapeHtml(t.name)}</div>
                                    <div class="text-xs text-slate-400">Modelo: <span class="text-slate-300 font-medium">${escapeHtml(t.model || 'N/A')}</span></div>
                                </div>
                                <div class="flex items-center gap-1.5">
                                    <span class="px-2 py-0.5 bg-amber-500/10 text-amber-400 border border-amber-500/20 rounded text-[11px] font-semibold whitespace-nowrap">${escapeHtml(t.status)}</span>
                                    <button onclick="deleteTraining(event, '${escapeHtml(t.name)}')" title="Excluir treinamento do disco" class="p-1 text-slate-500 hover:text-rose-400 transition">🗑️</button>
                                </div>
                            </div>
                            <a href="/training.html?id=${encodeURIComponent(t.name)}" class="text-xs text-amber-400 font-medium pt-1 flex items-center gap-1 group-hover:translate-x-1 transition-transform">
                                <span>Ver detalhes e métricas</span> &rarr;
                            </a>
                        </div>
                    `).join('');
                }

            } catch (err) {
                console.error(err);
                divC.innerHTML = `<p class="text-rose-400 text-xs p-3">Erro ao carregar origens: ${escapeHtml(err.message)}</p>`;
            }
        }

        async function deleteSource(event, sourceId) {
            if (event) event.stopPropagation();
            try {
                const impactRes = await fetch(`/api/deletion-impact/source/${encodeURIComponent(sourceId)}`);
                const impact = await impactRes.json();
                const deps = [...(impact.dependent_versions || []), ...(impact.dependent_trainings || [])];
                const sharedVideos = Object.keys(impact.shared_video_references || {});
                const cascade = deps.length > 0 && confirm(`A origem '${sourceId}' possui dependências:\n\nVersões: ${(impact.dependent_versions || []).join(', ') || 'nenhuma'}\nTreinamentos: ${(impact.dependent_trainings || []).join(', ') || 'nenhum'}\nVídeos também usados por outras origens: ${sharedVideos.length}\n\nOK: excluir também versões e treinamentos dependentes.\nCancelar: manter dependências, mesmo que fiquem inválidas.`);
                const deleteVideos = confirm(`Também apagar os arquivos de vídeo associados à origem?\n\n${sharedVideos.length ? `ATENÇÃO: ${sharedVideos.length} vídeo(s) também são referenciados por outras origens.` : 'Nenhum compartilhamento com outra origem foi detectado.'}\n\nOK: apagar vídeos.\nCancelar: preservar vídeos no disco.`);
                const typed = prompt(`Exclusão permanente. Digite exatamente '${sourceId}' para confirmar:`);
                if (typed !== sourceId) return;
                const res = await fetch(`/api/sources/${encodeURIComponent(sourceId)}?confirm=${encodeURIComponent(sourceId)}&cascade=${cascade}&delete_videos=${deleteVideos}`, { method: 'DELETE' });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || 'Erro ao excluir origem.');
                alert(data.message || 'Origem excluída com sucesso!');
                loadData();
            } catch (err) {
                alert(err.message);
            }
        }

        async function deleteRelease(event, versionId) {
            if (event) event.stopPropagation();
            try {
                const impactRes = await fetch(`/api/deletion-impact/version/${encodeURIComponent(versionId)}`);
                const impact = await impactRes.json();
                const cascade = (impact.dependent_trainings || []).length > 0 && confirm(`A versão '${versionId}' possui treinamentos dependentes:\n${(impact.dependent_trainings || []).join(', ')}\n\nOK: excluir também os treinamentos.\nCancelar: manter os treinamentos sem o dataset de origem.`);
                const typed = prompt(`Exclusão permanente. Digite exatamente '${versionId}' para confirmar:`);
                if (typed !== versionId) return;
                const res = await fetch(`/api/releases/${encodeURIComponent(versionId)}?confirm=${encodeURIComponent(versionId)}&cascade=${cascade}`, { method: 'DELETE' });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || 'Erro ao excluir release.');
                alert(data.message || 'Release excluída com sucesso!');
                loadData();
            } catch (err) {
                alert(err.message);
            }
        }

        async function deleteTraining(event, trainingId) {
            if (event) event.stopPropagation();
            const typed = prompt(`Os logs e pesos serão apagados permanentemente. Digite exatamente '${trainingId}' para confirmar:`);
            if (typed !== trainingId) return;
            try {
                const res = await fetch(`/api/trainings/${encodeURIComponent(trainingId)}?confirm=${encodeURIComponent(trainingId)}`, { method: 'DELETE' });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || 'Erro ao excluir treinamento.');
                alert(data.message || 'Treinamento excluído com sucesso!');
                loadData();
            } catch (err) {
                alert(err.message);
            }
        }

        function escapeHtml(text) {
            if (!text) return '';
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        loadData();
    </script>
</body>
</html>"""

    @app.get("/source.html", response_class=HTMLResponse)
    @app.get("/campaign.html", response_class=HTMLResponse)
    def source_detail_page():
        return """<!DOCTYPE html>
<html lang="pt-BR" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Studio - Detalhes da Origem</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background-color: #0f172a; color: #f8fafc; font-family: system-ui, -apple-system, sans-serif; }
    </style>
</head>
<body class="p-6 md:p-10">
    <div class="max-w-5xl mx-auto space-y-8">
        <!-- Header -->
        <header class="flex justify-between items-center pb-6 border-b border-slate-800">
            <div>
                <a href="/" class="text-xs text-indigo-400 hover:underline mb-1 inline-block">&larr; Voltar para a Tela Inicial</a>
                <h1 class="text-3xl font-extrabold tracking-tight text-indigo-400" id="camp-title">Origem de Dados</h1>
                <p class="text-slate-400 text-sm mt-1" id="camp-subtitle">Carregando informações da origem...</p>
            </div>
            <span id="camp-status-badge" class="px-3 py-1.5 bg-slate-800 text-indigo-400 border border-indigo-500/30 rounded-lg text-xs font-semibold">
                Status: ...
            </span>
        </header>

        <!-- Acordeão de Etapas -->
        <div class="space-y-4" id="accordion-container">

            <!-- Etapa 1: Seleção dos Vídeos -->
            <div class="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-lg transition" id="step-1-card">
                <button onclick="toggleStep(1)" class="w-full p-5 text-left flex justify-between items-center bg-slate-900 hover:bg-slate-800/60 transition">
                    <div class="flex items-center gap-3">
                        <span class="w-8 h-8 rounded-full bg-indigo-600/20 text-indigo-400 flex items-center justify-center font-bold text-sm border border-indigo-500/30">1</span>
                        <div>
                            <h3 class="font-bold text-slate-100 text-base">Etapa 1: Seleção dos Vídeos</h3>
                            <p class="text-xs text-slate-400">Vídeos associados a esta campanha</p>
                        </div>
                    </div>
                    <span id="step-1-status" class="text-xs font-semibold px-2.5 py-1 bg-emerald-500/10 text-emerald-400 rounded-md">✓ Concluído</span>
                </button>
                <div id="step-1-body" class="p-5 border-t border-slate-800 bg-slate-950/40 space-y-3">
                    <div id="step-1-videos" class="space-y-2 text-sm text-slate-300 font-mono">
                        Carregando lista de vídeos...
                    </div>
                </div>
            </div>

            <!-- Etapa 2: Modo de Extração dos Frames -->
            <div class="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-lg transition" id="step-2-card">
                <button onclick="toggleStep(2)" class="w-full p-5 text-left flex justify-between items-center bg-slate-900 hover:bg-slate-800/60 transition">
                    <div class="flex items-center gap-3">
                        <span class="w-8 h-8 rounded-full bg-indigo-600/20 text-indigo-400 flex items-center justify-center font-bold text-sm border border-indigo-500/30">2</span>
                        <div>
                            <h3 class="font-bold text-slate-100 text-base">Etapa 2: Seleção do Modo de Extração dos Frames</h3>
                            <p class="text-xs text-slate-400">Configuração de amostragem uniforme ou inteligente com modelo</p>
                        </div>
                    </div>
                    <span id="step-2-status" class="text-xs font-semibold px-2.5 py-1 bg-slate-800 text-slate-400 rounded-md">Pendente</span>
                </button>
                <div id="step-2-body" class="p-5 border-t border-slate-800 bg-slate-950/40 space-y-5">
                    <div id="step-2-locked-msg" class="hidden p-3.5 bg-emerald-500/10 border border-emerald-500/30 rounded-xl text-emerald-400 text-xs font-semibold flex items-center gap-2">
                        <span>🔒</span> Extração concluída. Esta etapa não pode mais ser alterada.
                    </div>

                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <!-- Modo Uniforme -->
                        <div onclick="selectExtractionMode('uniform')" id="card-mode-uniform" class="p-4 bg-slate-800/60 border-2 border-indigo-500 rounded-xl cursor-pointer hover:bg-slate-800 transition space-y-2">
                            <div class="flex justify-between items-center">
                                <span class="font-bold text-indigo-300 text-sm">Mode: Uniforme</span>
                                <span class="text-xs bg-indigo-500/20 text-indigo-300 px-2 py-0.5 rounded">Sem Filtro</span>
                            </div>
                            <p class="text-xs text-slate-300">
                                ℹ️ Extrai frames em um intervalo fixo de tempo (step) em todo o vídeo, sem carregar modelos. Ideal para amostragem limpa e imparcial.
                            </p>
                        </div>

                        <!-- Modo Inteligente -->
                        <div onclick="selectExtractionMode('smart')" id="card-mode-smart" class="p-4 bg-slate-800/30 border-2 border-slate-800 rounded-xl cursor-pointer hover:bg-slate-800 transition space-y-2">
                            <div class="flex justify-between items-center">
                                <span class="font-bold text-slate-200 text-sm">Mode: Inteligente</span>
                                <span class="text-xs bg-purple-500/20 text-purple-300 px-2 py-0.5 rounded">Com Modelo</span>
                            </div>
                            <p class="text-xs text-slate-300">
                                ℹ️ Usa um modelo pré-treinado em models/ para identificar trechos com objetos, extraindo mais frames onde há objetos e menos fora.
                            </p>
                        </div>
                    </div>

                    <!-- Formulário de Parâmetros -->
                    <div class="space-y-4 pt-2 border-t border-slate-800/60 text-sm">
                        <div id="uniform-opts">
                            <label class="block text-slate-300 font-medium mb-1">Intervalo de Frames (frame_step)</label>
                            <input type="number" id="extract-step" value="30" class="w-full md:w-1/2 bg-slate-800 border border-slate-700 rounded-lg p-2.5 text-white">
                            <span class="text-xs text-slate-400 block mt-1">Ex: 30 = 1 frame a cada 30 quadros (~1 segundo a 30fps)</span>
                        </div>

                        <div id="smart-opts" class="hidden space-y-3">
                            <div>
                                <label class="block text-slate-300 font-medium mb-1">Modelo Pré-Treinado (de models/)</label>
                                <select id="extract-model" class="w-full md:w-1/2 bg-slate-800 border border-slate-700 rounded-lg p-2.5 text-white">
                                    <option value="">Carregando modelos...</option>
                                </select>
                            </div>
                        </div>

                        <div>
                            <button id="btn-extract" onclick="executeExtraction()" class="px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs rounded-lg shadow-lg shadow-indigo-600/30 transition">
                                ▶ Executar Extração de Frames
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Etapa 3: Pré-Anotação -->
            <div class="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-lg transition" id="step-3-card">
                <button onclick="toggleStep(3)" class="w-full p-5 text-left flex justify-between items-center bg-slate-900 hover:bg-slate-800/60 transition">
                    <div class="flex items-center gap-3">
                        <span class="w-8 h-8 rounded-full bg-indigo-600/20 text-indigo-400 flex items-center justify-center font-bold text-sm border border-indigo-500/30">3</span>
                        <div>
                            <h3 class="font-bold text-slate-100 text-base">Etapa 3: Pré-Anotação e Geração do import_tasks.json</h3>
                            <p class="text-xs text-slate-400">Configuração de sugestões prévias para o Label Studio</p>
                        </div>
                    </div>
                    <span id="step-3-status" class="text-xs font-semibold px-2.5 py-1 bg-slate-800 text-slate-400 rounded-md">Pendente</span>
                </button>
                <div id="step-3-body" class="p-5 border-t border-slate-800 bg-slate-950/40 space-y-4 hidden">
                    <div id="step-3-locked-msg" class="hidden p-3.5 bg-emerald-500/10 border border-emerald-500/30 rounded-xl text-emerald-400 text-xs font-semibold flex items-center gap-2">
                        <span>🔒</span> Pré-anotação concluída. Esta etapa não pode mais ser alterada.
                    </div>

                    <p class="text-xs text-slate-300">
                        Nesta etapa, você decide se deseja usar um modelo para gerar pré-anotações automáticas nas imagens ou se deseja pular essa etapa (nenhuma imagem será anotada previamente).
                    </p>
                    
                    <div class="space-y-3 text-sm">
                        <div class="flex items-center gap-3">
                            <input type="radio" id="pre-none" name="pre_opt" value="none" checked onchange="togglePreAnnotateOpt()" class="w-4 h-4 text-indigo-600">
                            <label for="pre-none" class="text-slate-200 font-medium">Pular Pré-Anotação (Rotulação 100% manual limpa)</label>
                        </div>
                        <div class="flex items-center gap-3">
                            <input type="radio" id="pre-model" name="pre_opt" value="model" onchange="togglePreAnnotateOpt()" class="w-4 h-4 text-indigo-600">
                            <label for="pre-model" class="text-slate-200 font-medium">Usar Modelo de Detecção para Pré-Anotar</label>
                        </div>

                        <div id="pre-model-selector" class="pl-7 hidden space-y-2">
                            <label class="block text-slate-300 text-xs font-medium">Selecione o Modelo para Pré-Anotação</label>
                            <select id="pre-model-dropdown" class="w-full md:w-1/2 bg-slate-800 border border-slate-700 rounded-lg p-2 text-white text-xs">
                                <option value="">Carregando...</option>
                            </select>
                        </div>

                        <div class="pt-2">
                            <button id="btn-build-import" onclick="executeBuildImport()" class="px-5 py-2.5 bg-amber-600 hover:bg-amber-500 text-white font-medium text-xs rounded-lg shadow-lg shadow-amber-600/30 transition">
                                ▶ Gerar import_tasks.json
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Etapa 4: Instruções, ML Backend e Monitoramento de finished_tasks -->
            <div class="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden shadow-lg transition" id="step-4-card">
                <button onclick="toggleStep(4)" class="w-full p-5 text-left flex justify-between items-center bg-slate-900 hover:bg-slate-800/60 transition">
                    <div class="flex items-center gap-3">
                        <span class="w-8 h-8 rounded-full bg-indigo-600/20 text-indigo-400 flex items-center justify-center font-bold text-sm border border-indigo-500/30">4</span>
                        <div>
                            <h3 class="font-bold text-slate-100 text-base">Etapa 4: Label Studio, ML Backend & Exportação</h3>
                            <p class="text-xs text-slate-400">Servidor de detecções e detecção automática do JSON final</p>
                        </div>
                    </div>
                    <span id="step-4-status" class="text-xs font-semibold px-2.5 py-1 bg-slate-800 text-slate-400 rounded-md">Pendente</span>
                </button>
                <div id="step-4-body" class="p-5 border-t border-slate-800 bg-slate-950/40 space-y-5 text-sm hidden">
                    
                    <!-- Opções do Servidor -->
                    <div class="bg-slate-900 border border-slate-800 p-4 rounded-xl space-y-3">
                        <h4 class="font-bold text-indigo-300 text-sm">🚀 Servidor de Detecção (ML Backend) + Label Studio</h4>

                        <div id="ls-integration-panel" class="p-3.5 bg-slate-950/70 border border-slate-700 rounded-xl space-y-3">
                            <div class="flex flex-col md:flex-row md:items-center justify-between gap-2">
                                <div>
                                    <div class="flex items-center gap-2">
                                        <span class="font-semibold text-slate-200 text-xs">Integração automática</span>
                                        <span id="ls-integration-status" class="px-2 py-0.5 rounded bg-slate-800 text-slate-400 text-[10px] font-bold">Verificando</span>
                                    </div>
                                    <p id="ls-integration-message" class="text-[11px] text-slate-400 mt-1">
                                        Verificando vínculo com o Label Studio...
                                    </p>
                                </div>
                                <button type="button" onclick="toggleLabelStudioSettings()" class="text-[11px] text-indigo-300 hover:text-indigo-200">
                                    Configurar conexão
                                </button>
                            </div>
                            <div id="ls-integration-details" class="hidden text-[11px] text-slate-400"></div>
                            <button id="ls-confirm-partial" type="button" onclick="confirmPartialPredictionCoverage()" class="hidden px-3 py-1.5 bg-amber-600 hover:bg-amber-500 text-white text-[11px] font-medium rounded-lg">
                                Continuar conscientemente com cobertura parcial
                            </button>
                            <div id="ls-settings-form" class="hidden border-t border-slate-800 pt-3 space-y-2">
                                <p class="text-[11px] text-slate-400">
                                    Esta configuração é feita uma única vez por computador. O Dataset Studio usará a API oficial para criar, importar e configurar os próximos projetos automaticamente.
                                </p>
                                <label class="block text-[11px] text-slate-300">URL do Label Studio</label>
                                <input id="ls-base-url" value="http://127.0.0.1:8080" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2 text-white text-xs">
                                <label class="block text-[11px] text-slate-300">Token de acesso</label>
                                <input id="ls-api-key" type="password" autocomplete="off" placeholder="Cole o token obtido em Account & Settings" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2 text-white text-xs">
                                <div class="flex flex-wrap gap-2 pt-1">
                                    <button type="button" onclick="saveLabelStudioSettings()" class="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white text-[11px] font-medium rounded-lg">
                                        Salvar e preparar esta origem
                                    </button>
                                    <button type="button" onclick="toggleLabelStudioSettings()" class="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 text-[11px] rounded-lg">
                                        Fechar
                                    </button>
                                </div>
                            </div>
                        </div>
                        
                        <div class="flex items-center gap-2">
                            <input type="checkbox" id="enable-ml-backend" onchange="toggleMlBackendOpts()" class="w-4 h-4 rounded bg-slate-800 text-indigo-600">
                            <label for="enable-ml-backend" class="text-slate-200 text-xs font-medium">Iniciar Servidor de Detecção em Segundo Plano (ML Backend na porta 9090)</label>
                        </div>

                        <div id="ml-backend-opts" class="hidden pl-6 space-y-2">
                            <label class="block text-xs text-slate-400">Modelo do Servidor de Detecção:</label>
                            <select id="ml-backend-model" class="w-full md:w-1/2 bg-slate-800 border border-slate-700 rounded-lg p-2 text-white text-xs">
                                <option value="">Carregando modelos...</option>
                            </select>
                        </div>

                        <div>
                            <button id="btn-start-label-studio" onclick="startLabelStudioService()" class="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs rounded-lg shadow-md shadow-indigo-600/30 transition">
                                🚀 Iniciar Label Studio (+ ML Backend)
                            </button>
                        </div>
                    </div>

                    <!-- Instrução para finished_tasks -->
                    <div class="p-4 bg-indigo-500/10 border border-indigo-500/20 rounded-xl space-y-2 text-xs">
                        <h4 class="font-bold text-indigo-300 text-sm">📁 Salvar Exportação no Diretório:</h4>
                        <p class="text-slate-300">
                            Após rotular no Label Studio, salve ou exporte o arquivo JSON final na pasta:
                        </p>
                        <div class="p-2.5 bg-slate-950 rounded border border-slate-800 font-mono text-indigo-400 select-all" id="finished-tasks-path">
                            ...
                        </div>
                        <p class="text-slate-400">
                            O Dataset Studio monitora esta pasta automaticamente. Assim que o arquivo for salvo lá, a etapa será concluída!
                        </p>
                    </div>

                    <!-- Container de Painéis Individuais para Cada Exportação JSON Detectada -->
                    <div id="accepted-revisions-container" class="space-y-2"></div>
                    <div id="finished-exports-container" class="hidden space-y-4">
                        <!-- Painéis gerados dinamicamente via JS -->
                    </div>

                </div>
            </div>

        </div>
    </div>

    <script>
        const urlParams = new URLSearchParams(window.location.search);
        const campaignId = urlParams.get('id');
        let selectedMode = 'uniform';
        let step2Locked = false;
        let step3Locked = false;

        function escapeHtml(text) {
            if (!text) return '';
            return String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
        }

        function toggleStep(stepNum) {
            const body = document.getElementById(`step-${stepNum}-body`);
            body.classList.toggle('hidden');
        }

        function toggleLabelStudioSettings(forceOpen = null) {
            const form = document.getElementById('ls-settings-form');
            if (!form) return;
            const open = forceOpen === null ? form.classList.contains('hidden') : forceOpen;
            form.classList.toggle('hidden', !open);
        }

        function renderLabelStudioIntegration(info) {
            if (!info) return;
            const badge = document.getElementById('ls-integration-status');
            const message = document.getElementById('ls-integration-message');
            const details = document.getElementById('ls-integration-details');
            const partialButton = document.getElementById('ls-confirm-partial');
            const credentials = info.credentials || {};
            const integration = info.integration || (info.project_id ? info : null);
            const plan = info.prediction_plan || (integration && integration.prediction_coverage) || null;
            const status = info.status || (integration ? 'ready' : 'unknown');
            const styles = {
                'ready': 'bg-emerald-500/10 text-emerald-400',
                'ready-to-prepare': 'bg-indigo-500/10 text-indigo-300',
                'needs-token': 'bg-amber-500/10 text-amber-300',
                'partial-predictions': 'bg-amber-500/10 text-amber-300',
                'waiting-import': 'bg-slate-800 text-slate-400',
            };
            const labels = {
                'ready': 'Pronta',
                'ready-to-prepare': 'Pronta para preparar',
                'needs-token': 'Configuração única',
                'partial-predictions': 'Requer confirmação',
                'waiting-import': 'Aguardando tarefas',
            };
            badge.className = `px-2 py-0.5 rounded text-[10px] font-bold ${styles[status] || 'bg-rose-500/10 text-rose-300'}`;
            badge.innerText = labels[status] || 'Atenção';
            message.innerText = info.message || (
                status === 'ready'
                    ? 'Projeto vinculado e configurações verificadas automaticamente.'
                    : 'Verifique a configuração da integração.'
            );
            if (credentials.base_url) {
                document.getElementById('ls-base-url').value = credentials.base_url;
            } else if (integration && integration.base_url) {
                document.getElementById('ls-base-url').value = integration.base_url;
            }
            const detailParts = [];
            const projectId = info.project_id || (integration && integration.project_id);
            if (projectId) detailParts.push(`Projeto: ${projectId}`);
            if (plan && plan.uses_predictions) {
                detailParts.push(`Predição: ${plan.covered_tasks}/${plan.total_tasks} tarefas`);
                if (plan.selected_version) detailParts.push(`Versão: ${plan.selected_version}`);
            } else if (plan) {
                detailParts.push('Rotulação manual sem predições');
            }
            details.innerText = detailParts.join(' • ');
            details.classList.toggle('hidden', detailParts.length === 0);
            partialButton.classList.toggle('hidden', status !== 'partial-predictions');
            if (status === 'needs-token') toggleLabelStudioSettings(true);
        }

        async function loadLabelStudioIntegration() {
            if (!campaignId) return;
            try {
                const res = await fetch(`/api/sources/${encodeURIComponent(campaignId)}/label-studio`);
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || 'Falha ao verificar integração.');
                renderLabelStudioIntegration(data);
            } catch (err) {
                renderLabelStudioIntegration({status: 'error', message: err.message});
            }
        }

        async function prepareLabelStudioProject(allowPartial = false) {
            const res = await fetch(`/api/sources/${encodeURIComponent(campaignId)}/label-studio/prepare`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({allow_partial_predictions: allowPartial})
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Falha ao preparar o projeto.');
            renderLabelStudioIntegration(data);
            return data;
        }

        async function confirmPartialPredictionCoverage() {
            const confirmed = confirm(
                'Algumas tarefas serão abertas sem preanotações. Você poderá desenhar as caixas manualmente. Deseja continuar?'
            );
            if (!confirmed) return;
            try {
                const prepared = await prepareLabelStudioProject(true);
                alert(`Projeto ${prepared.project_id} preparado com a cobertura parcial informada.`);
            } catch (err) {
                alert(err.message);
            }
        }

        async function saveLabelStudioSettings() {
            const baseUrl = document.getElementById('ls-base-url').value.trim();
            const apiKey = document.getElementById('ls-api-key').value.trim();
            if (!apiKey) {
                alert('Informe o token de acesso do Label Studio.');
                return;
            }
            try {
                const res = await fetch('/api/label-studio/settings', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({base_url: baseUrl, api_key: apiKey})
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || 'Não foi possível validar o token.');
                document.getElementById('ls-api-key').value = '';
                const prepared = await prepareLabelStudioProject(false);
                toggleLabelStudioSettings(false);
                alert(`Integração configurada. Projeto ${prepared.project_id} pronto para rotulação.`);
            } catch (err) {
                alert(err.message);
            }
        }

        function selectExtractionMode(mode) {
            if (step2Locked) return;
            selectedMode = mode;
            const cardU = document.getElementById('card-mode-uniform');
            const cardS = document.getElementById('card-mode-smart');
            const optsU = document.getElementById('uniform-opts');
            const optsS = document.getElementById('smart-opts');

            if (mode === 'uniform') {
                cardU.className = "p-4 bg-slate-800/60 border-2 border-indigo-500 rounded-xl cursor-pointer hover:bg-slate-800 transition space-y-2";
                cardS.className = "p-4 bg-slate-800/30 border-2 border-slate-800 rounded-xl cursor-pointer hover:bg-slate-800 transition space-y-2";
                optsU.classList.remove('hidden');
                optsS.classList.add('hidden');
            } else {
                cardS.className = "p-4 bg-slate-800/60 border-2 border-purple-500 rounded-xl cursor-pointer hover:bg-slate-800 transition space-y-2";
                cardU.className = "p-4 bg-slate-800/30 border-2 border-slate-800 rounded-xl cursor-pointer hover:bg-slate-800 transition space-y-2";
                optsS.classList.remove('hidden');
                optsU.classList.add('hidden');
            }
        }

        function togglePreAnnotateOpt() {
            if (step3Locked) return;
            const val = document.querySelector('input[name="pre_opt"]:checked').value;
            const sel = document.getElementById('pre-model-selector');
            if (val === 'model') {
                sel.classList.remove('hidden');
            } else {
                sel.classList.add('hidden');
            }
        }

        function toggleMlBackendOpts() {
            const chk = document.getElementById('enable-ml-backend').checked;
            const opts = document.getElementById('ml-backend-opts');
            if (chk) opts.classList.remove('hidden');
            else opts.classList.add('hidden');
        }

        async function executeExtraction() {
            if (step2Locked) {
                alert('A extração de frames já foi concluída e esta etapa não pode mais ser alterada.');
                return;
            }
            try {
                const model = document.getElementById('extract-model').value;
                const payload = {
                    mode: selectedMode,
                    uniform_frame_step: Number(document.getElementById('extract-step').value || 30),
                    model: selectedMode === 'smart' ? model : null
                };
                if (selectedMode === 'smart' && !model) {
                    throw new Error('Selecione um modelo para a extração inteligente.');
                }
                const res = await fetch(`/api/campaigns/${campaignId}/extract`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || 'Erro na extração');
                alert('Extração concluída com sucesso!');
                loadCampaignDetails();
            } catch (err) {
                alert(err.message);
            }
        }

        async function executeBuildImport() {
            if (step3Locked) {
                alert('A etapa de pré-anotação já foi concluída e não pode mais ser alterada.');
                return;
            }
            try {
                const selected = document.querySelector('input[name="pre_opt"]:checked').value;
                const model = document.getElementById('pre-model-dropdown').value;
                if (selected === 'model' && !model) throw new Error('Selecione um modelo para pré-anotar.');
                const res = await fetch(`/api/campaigns/${campaignId}/import-tasks`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({mode: selected, model: selected === 'model' ? model : null})
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || 'Erro ao gerar import tasks');
                alert('Arquivo import_tasks.json gerado com sucesso!');
                loadCampaignDetails();
            } catch (err) {
                alert(err.message);
            }
        }

        async function deleteRevision(revisionId) {
            const impactRes = await fetch(`/api/deletion-impact/revision/${encodeURIComponent(revisionId)}?source_id=${encodeURIComponent(campaignId)}`);
            const impact = await impactRes.json();
            const deps = [...(impact.dependent_versions || []), ...(impact.dependent_trainings || [])];
            const cascade = deps.length > 0 && confirm(`A revisão '${revisionId}' é usada por:\nVersões: ${(impact.dependent_versions || []).join(', ') || 'nenhuma'}\nTreinamentos: ${(impact.dependent_trainings || []).join(', ') || 'nenhum'}\n\nOK: excluir em cascata.\nCancelar: manter dependências potencialmente inválidas.`);
            const typed = prompt(`Digite exatamente '${revisionId}' para excluir permanentemente a revisão:`);
            if (typed !== revisionId) return;
            const res = await fetch(`/api/sources/${encodeURIComponent(campaignId)}/revisions/${encodeURIComponent(revisionId)}?confirm=${encodeURIComponent(revisionId)}&cascade=${cascade}`, {method: 'DELETE'});
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || 'Erro ao excluir revisão.');
            alert(data.message);
            loadCampaignDetails();
        }

        async function startLabelStudioService() {
            const btn = document.getElementById('btn-start-label-studio');
            const chkMl = document.getElementById('enable-ml-backend').checked;
            const selModel = document.getElementById('ml-backend-model').value;

            const originalText = btn ? btn.innerText : '🚀 Iniciar Label Studio (+ ML Backend)';
            if (btn) {
                btn.disabled = true;
                btn.innerText = '⏳ Aguardando servidor subir...';
            }

            // Abrir a página imediatamente em uma nova aba para evitar bloqueios de popup do navegador
            const labelStudioTab = window.open('about:blank', '_blank');
            if (labelStudioTab) {
                labelStudioTab.document.write(`
                    <!DOCTYPE html>
                    <html lang="pt-BR" style="background:#0f172a;color:#f8fafc;font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;">
                    <head><title>Carregando Label Studio...</title></head>
                    <body style="text-align:center;">
                        <h2 style="color:#818cf8;">🚀 Iniciando servidor do Label Studio...</h2>
                        <p style="color:#94a3b8;">A página será aberta automaticamente assim que o servidor estiver pronto.</p>
                    </body>
                    </html>
                `);
            }

            try {
                const res = await fetch(`/api/campaigns/${campaignId}/start-label-studio`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ enable_ml: chkMl, model: selModel })
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || 'Erro ao iniciar servidor do Label Studio.');
                renderLabelStudioIntegration(data.integration);

                if (data.url) {
                    if (labelStudioTab) {
                        labelStudioTab.location.href = data.url;
                    } else {
                        window.open(data.url, '_blank');
                    }
                }

                if (data.integration && data.integration.status === 'needs-token') {
                    alert(
                        'O Label Studio foi iniciado. Faça login, copie uma única vez o token em ' +
                        'Account & Settings > Access Token e cole no painel de integração do Dataset Studio.'
                    );
                }

                if (btn) {
                    btn.disabled = false;
                    btn.innerText = '🚀 Abrir Label Studio (Servidor Ativo)';
                    btn.className = "px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-xs rounded-lg shadow-md shadow-emerald-600/30 transition";
                }
            } catch (err) {
                if (labelStudioTab) labelStudioTab.close();
                if (btn) {
                    btn.disabled = false;
                    btn.innerText = originalText;
                }
                alert(err.message);
            }
        }

        async function acceptAndCreateReleaseForExport(selectedPath, exportName) {
            if (!selectedPath) return;

            try {
                const res = await fetch(`/api/campaigns/${campaignId}/accept-export`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ path: selectedPath, allow_pending: true })
                });
                const contentType = res.headers.get('content-type') || '';
                let data = {};
                if (contentType.includes('application/json')) {
                    data = await res.json();
                } else {
                    const text = await res.text();
                    throw new Error(`Erro do servidor (${res.status}): ${text.slice(0, 150)}`);
                }
                if (!res.ok) throw new Error(data.detail || 'Erro ao aceitar o JSON selecionado.');
                
                const cleanName = exportName ? exportName.replace(new RegExp('\\.json$', 'i'), '') : 'release';
                const relId = `release_${campaignId}_${cleanName}`;
                const revisionId = data.revision_id;
                if (!revisionId) throw new Error('O servidor não informou a revisão criada para este JSON.');
                window.location.href = `/release.html?campaign=${encodeURIComponent(campaignId)}&id=${encodeURIComponent(relId)}&revision_id=${encodeURIComponent(revisionId)}`;
            } catch (err) {
                alert(err.message);
            }
        }

        function createReleaseFromCampaign() {
            const relId = `release_${campaignId}`;
            window.location.href = `/release.html?campaign=${campaignId}&id=${relId}`;
        }

        async function loadCampaignDetails() {
            if (!campaignId) {
                alert('ID da campanha não informado.');
                window.location.href = '/';
                return;
            }

            try {
                const resSt = await fetch(`/api/campaigns/${campaignId}`);
                const st = await resSt.json();

                document.getElementById('camp-title').innerText = st.campaign_id;
                document.getElementById('camp-subtitle').innerText = `Vídeos: ${st.videos} | Frames Extraídos: ${st.frames} | Import Tasks: ${st.import_tasks}`;
                document.getElementById('camp-status-badge').innerText = `Etapa: ${st.next_action}`;

                // Etapa 1 - Lista Detalhada de Vídeos
                if (st.video_details && st.video_details.length > 0) {
                    const listHtml = `
                        <div class="text-xs text-emerald-400 font-semibold mb-3">
                            ✓ ${st.videos} arquivo(s) de vídeo associado(s) à campanha:
                        </div>
                        <div class="overflow-x-auto">
                            <table class="w-full text-left text-xs text-slate-300 font-sans border border-slate-800/80 rounded-lg overflow-hidden">
                                <thead class="bg-slate-800/80 text-slate-400 uppercase text-[10px] font-mono">
                                    <tr>
                                        <th class="p-2.5 border-b border-slate-800">Arquivo</th>
                                        <th class="p-2.5 border-b border-slate-800">Tamanho</th>
                                        <th class="p-2.5 border-b border-slate-800">Resolução</th>
                                        <th class="p-2.5 border-b border-slate-800">Framerate (FPS)</th>
                                    </tr>
                                </thead>
                                <tbody class="divide-y divide-slate-800/60 font-mono">
                                    ${st.video_details.map(v => `
                                        <tr class="hover:bg-slate-800/40 transition">
                                            <td class="p-2.5 font-medium text-slate-200">${escapeHtml(v.name)}</td>
                                            <td class="p-2.5 text-slate-400">${escapeHtml(v.size_human)}</td>
                                            <td class="p-2.5 text-indigo-300">${escapeHtml(v.resolution)}</td>
                                            <td class="p-2.5 text-slate-300">${v.fps > 0 ? v.fps + ' fps' : 'N/A'}</td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    `;
                    document.getElementById('step-1-videos').innerHTML = listHtml;
                } else {
                    document.getElementById('step-1-videos').innerText = `✓ ${st.videos} arquivo(s) de vídeo associado(s) à campanha.`;
                }

                // Etapa 2 & 3 (Bloqueio ao concluir)
                const revisionsContainer = document.getElementById('accepted-revisions-container');
                const revisions = st.annotation_revisions || [];
                revisionsContainer.innerHTML = revisions.length ? `
                    <div class="text-xs font-semibold text-slate-300">Revisões aceitas</div>
                    ${revisions.map(rev => `<div class="flex items-center justify-between p-2.5 bg-slate-800/50 border border-slate-700/50 rounded-lg"><span class="font-mono text-xs text-purple-300">${escapeHtml(rev.revision_id)}</span><button onclick="deleteRevision('${encodeURIComponent(rev.revision_id)}'.startsWith('%') ? decodeURIComponent('${encodeURIComponent(rev.revision_id)}') : '${escapeHtml(rev.revision_id)}')" class="text-xs text-rose-400 hover:text-rose-300">Excluir</button></div>`).join('')}
                ` : '';
                const extraction = st.extraction || {};
                selectedMode = extraction.mode || 'uniform';
                selectExtractionMode(selectedMode);
                if (extraction.uniform_frame_step || extraction.frame_step) {
                    document.getElementById('extract-step').value = extraction.uniform_frame_step || extraction.frame_step;
                }
                if (extraction.model) {
                    const modelSelect = document.getElementById('extract-model');
                    modelSelect.value = extraction.model;
                }
                if (st.frames > 0) {
                    step2Locked = true;
                    document.getElementById('step-2-status').className = "text-xs font-semibold px-2.5 py-1 bg-emerald-500/10 text-emerald-400 rounded-md";
                    document.getElementById('step-2-status').innerText = "✓ Concluído";
                    document.getElementById('step-3-body').classList.remove('hidden');

                    // Desabilitar controles da Etapa 2
                    document.getElementById('extract-step').disabled = true;
                    document.getElementById('extract-model').disabled = true;

                    const cardU = document.getElementById('card-mode-uniform');
                    const cardS = document.getElementById('card-mode-smart');
                    if (cardU) {
                        cardU.onclick = null;
                        cardU.classList.remove('cursor-pointer');
                        cardU.classList.add('cursor-not-allowed', 'opacity-75');
                    }
                    if (cardS) {
                        cardS.onclick = null;
                        cardS.classList.remove('cursor-pointer');
                        cardS.classList.add('cursor-not-allowed', 'opacity-75');
                    }

                    const btnExtract = document.getElementById('btn-extract');
                    if (btnExtract) {
                        btnExtract.disabled = true;
                        btnExtract.className = "px-5 py-2.5 bg-slate-800 text-slate-500 font-medium text-xs rounded-lg cursor-not-allowed border border-slate-700/50";
                        btnExtract.innerText = "🔒 Extração Concluída";
                    }
                    const msg2 = document.getElementById('step-2-locked-msg');
                    if (msg2) msg2.classList.remove('hidden');
                }

                if (st.import_tasks > 0) {
                    step3Locked = true;
                    document.getElementById('step-3-status').className = "text-xs font-semibold px-2.5 py-1 bg-emerald-500/10 text-emerald-400 rounded-md";
                    document.getElementById('step-3-status').innerText = "✓ Concluído";
                    document.getElementById('step-4-body').classList.remove('hidden');

                    // Desabilitar controles da Etapa 3
                    document.getElementById('pre-none').disabled = true;
                    document.getElementById('pre-model').disabled = true;
                    document.getElementById('pre-model-dropdown').disabled = true;

                    const btnBuild = document.getElementById('btn-build-import');
                    if (btnBuild) {
                        btnBuild.disabled = true;
                        btnBuild.className = "px-5 py-2.5 bg-slate-800 text-slate-500 font-medium text-xs rounded-lg cursor-not-allowed border border-slate-700/50";
                        btnBuild.innerText = "🔒 import_tasks.json Gerado";
                    }
                    const msg3 = document.getElementById('step-3-locked-msg');
                    if (msg3) msg3.classList.remove('hidden');
                    loadLabelStudioIntegration();
                }

                // finished_tasks info
                const fin = st.finished_info || {};
                document.getElementById('finished-tasks-path').innerText = fin.finished_tasks_dir || `campaigns/${campaignId}/label_studio/finished_tasks`;

                const container = document.getElementById('finished-exports-container');
                if (fin.found && fin.exports && fin.exports.length > 0) {
                    document.getElementById('step-4-status').className = "text-xs font-semibold px-2.5 py-1 bg-emerald-500/10 text-emerald-400 rounded-md";
                    document.getElementById('step-4-status').innerText = "✓ Concluído";
                    container.classList.remove('hidden');

                    let panelsHtml = '';
                    fin.exports.forEach(exp => {
                        const m = exp.metrics || {};
                        const clsEntries = Object.entries(m.class_counts || {});
                        const clsText = clsEntries.length > 0
                            ? clsEntries.map(([k, v]) => `${k}: ${v}`).join(', ')
                            : 'Nenhuma caixa registrada';

                        panelsHtml += `
                            <div class="bg-slate-900 border border-slate-800 hover:border-emerald-500/50 transition rounded-xl p-4 space-y-4">
                                <div class="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-slate-800/80 pb-3">
                                    <div class="flex items-center gap-2.5">
                                        <span class="text-base">📄</span>
                                        <div>
                                            <h4 class="font-bold text-slate-100 text-sm font-mono">${escapeHtml(exp.name)}</h4>
                                            <p class="text-[11px] text-slate-400">Arquivo JSON de exportação do Label Studio</p>
                                        </div>
                                    </div>
                                    <div>
                                        <button onclick="acceptAndCreateReleaseForExport(decodeURIComponent('${encodeURIComponent(exp.path)}'), decodeURIComponent('${encodeURIComponent(exp.name)}'))" class="px-4 py-2 bg-purple-600 hover:bg-purple-500 text-white font-medium text-xs rounded-lg shadow-md shadow-purple-600/30 transition flex items-center gap-1.5 whitespace-nowrap">
                                            <span>📦 Criar Release com este JSON</span>
                                            <span>&rarr;</span>
                                        </button>
                                    </div>
                                </div>

                                <!-- Cards de Métricas em Tempo Real -->
                                <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3 text-center">
                                    <div class="p-3 bg-slate-800/50 rounded-lg border border-slate-700/50">
                                        <div class="text-slate-400 text-[11px] font-medium">📷 Total Imagens</div>
                                        <div class="text-lg font-bold text-indigo-300 font-mono mt-1">${m.total_tasks || 0}</div>
                                    </div>
                                    <div class="p-3 bg-slate-800/50 rounded-lg border border-emerald-500/30">
                                        <div class="text-emerald-400 text-[11px] font-medium">✓ Anotadas</div>
                                        <div class="text-lg font-bold text-emerald-300 font-mono mt-1">${m.tasks_completed || 0}</div>
                                    </div>
                                    <div class="p-3 bg-slate-800/50 rounded-lg border border-amber-500/30">
                                        <div class="text-amber-400 text-[11px] font-medium">⏳ Não Anotadas</div>
                                        <div class="text-lg font-bold text-amber-300 font-mono mt-1">${m.tasks_deferred || 0}</div>
                                    </div>
                                    <div class="p-3 bg-slate-800/50 rounded-lg border border-rose-500/30">
                                        <div class="text-rose-400 text-[11px] font-medium">🚫 Canceladas</div>
                                        <div class="text-lg font-bold text-rose-300 font-mono mt-1">${m.tasks_cancelled || 0}</div>
                                    </div>
                                    <div class="p-3 bg-slate-800/50 rounded-lg border border-slate-700/50">
                                        <div class="text-slate-400 text-[11px] font-medium">📦 Caixas (Boxes)</div>
                                        <div class="text-lg font-bold text-indigo-200 font-mono mt-1">${m.total_boxes || 0}</div>
                                    </div>
                                    <div class="p-3 bg-slate-800/50 rounded-lg border border-slate-700/50">
                                        <div class="text-slate-400 text-[11px] font-medium">⚪ Negativos</div>
                                        <div class="text-lg font-bold text-slate-300 font-mono mt-1">${m.confirmed_negatives || 0}</div>
                                    </div>
                                </div>

                                <div class="p-3 bg-slate-800/30 rounded-lg border border-slate-800 flex items-center justify-between text-xs">
                                    <span class="text-slate-400 font-medium">🏷️ Contagem por Classe:</span>
                                    <span class="font-bold text-purple-300 font-mono">${escapeHtml(clsText)}</span>
                                </div>
                            </div>
                        `;
                    });
                    container.innerHTML = panelsHtml;
                } else {
                    container.classList.add('hidden');
                }

                const activeRev = st.latest_annotation_revision || (st.annotation_revisions && st.annotation_revisions.length > 0 ? st.annotation_revisions[st.annotation_revisions.length - 1].revision_id : null);
                const activeBadge = document.getElementById('active-revision-badge');
                if (activeBadge) {
                    activeBadge.innerText = activeRev ? `Revisão Ativa para Release: ${activeRev}` : '';
                }

                // Carregar Modelos para dropdowns
                const resM = await fetch('/api/models');
                const models = await resM.json();
                const selM = document.getElementById('extract-model');
                const selPre = document.getElementById('pre-model-dropdown');
                const selMl = document.getElementById('ml-backend-model');

                if (models.length > 0) {
                    const optsHtml = models.map(m => `<option value="${m}">${m}</option>`).join('');
                    selM.innerHTML = optsHtml;
                    selPre.innerHTML = optsHtml;
                    selMl.innerHTML = optsHtml;
                } else {
                    const empty = '<option value="">Nenhum modelo .pt em models/</option>';
                    selM.innerHTML = empty;
                    selPre.innerHTML = empty;
                    selMl.innerHTML = empty;
                }

                if (st.extraction && st.extraction.model) {
                    selM.value = st.extraction.model;
                }
                if (st.annotation_backend === 'local' || st.annotation_model) {
                    document.getElementById('pre-model').checked = true;
                    document.getElementById('pre-none').checked = false;
                    document.getElementById('pre-model-selector').classList.remove('hidden');
                    if (st.annotation_model) {
                        selPre.value = st.annotation_model;
                        selMl.value = st.annotation_model;
                    }
                } else {
                    document.getElementById('pre-none').checked = true;
                    document.getElementById('pre-model').checked = false;
                }

            } catch (err) {
                console.error(err);
            }
        }

        loadCampaignDetails();
    </script>
</body>
</html>"""

    @app.get("/version.html", response_class=HTMLResponse)
    @app.get("/release.html", response_class=HTMLResponse)
    def version_detail_page():
        return """<!DOCTYPE html>
<html lang="pt-BR" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Studio - Detalhes da Versão & Treinamento</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background-color: #0f172a; color: #f8fafc; font-family: system-ui, -apple-system, sans-serif; }
    </style>
</head>
<body class="p-6 md:p-10">
    <div class="max-w-5xl mx-auto space-y-8">
        <!-- Header -->
        <header class="flex justify-between items-center pb-6 border-b border-slate-800">
            <div>
                <a href="/" class="text-xs text-indigo-400 hover:underline mb-1 inline-block">&larr; Voltar para a Tela Inicial</a>
                <h1 class="text-3xl font-extrabold tracking-tight text-emerald-400" id="rel-title">Versão do Dataset</h1>
                <p class="text-slate-400 text-sm mt-1" id="rel-subtitle">Divisão por unidades experimentais completas, sem vazamento entre splits</p>
            </div>
            <span id="rel-status-badge" class="px-3 py-1.5 bg-slate-800 text-emerald-400 border border-emerald-500/30 rounded-lg text-xs font-semibold">
                Status: ...
            </span>
        </header>
        <div id="release-revision-notice" class="px-4 py-3 bg-purple-500/10 border border-purple-500/30 rounded-xl text-xs text-purple-200">
            Revisão de anotação: <span id="release-revision-id" class="font-mono font-bold">carregando...</span>
        </div>

        <!-- Seção 1: Divisão dos Vídeos e Calculadora de Splits -->
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
            <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-4">
                <h2 class="text-xl font-bold text-slate-100 flex items-center gap-2">
                    <span>🎥</span> Atribuição dos Vídeos ao Dataset
                </h2>
                <div class="flex items-center gap-2">
                    <label class="text-xs text-slate-400 font-medium whitespace-nowrap">Nome/ID da Release:</label>
                    <input type="text" id="input-release-id" class="bg-slate-800 border border-slate-700 rounded-lg p-2 text-xs text-emerald-300 font-mono w-64 focus:border-emerald-500 focus:outline-none font-bold">
                    <select id="evaluation-level" onchange="updateSplitPreview()" class="bg-slate-800 border border-slate-700 rounded-lg p-2 text-xs text-white">
                        <option value="standard">Padrão: treino + validação + teste</option>
                        <option value="pilot">Piloto: permite apenas treino</option>
                        <option value="robust">Robusto: inclui teste de estresse</option>
                    </select>
                </div>
            </div>
            <div class="p-3 bg-amber-500/10 border border-amber-500/30 rounded-xl text-xs text-amber-200">
                Uma versão padrão exige unidades experimentais independentes em treino, validação e teste normal.
                O modo piloto permite começar com uma única unidade, mas suas métricas não comprovam generalização.
                O teste de estresse é opcional, salvo no modo robusto.
            </div>
            <div id="split-quality-assessment" class="p-3 bg-slate-950/60 border border-slate-800 rounded-xl text-xs text-slate-400">
                Atribua as unidades para verificar suficiência de frames e anotações.
            </div>

            <!-- Cards de Métricas em Tempo Real (4 Splits) -->
            <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
                <div class="p-4 bg-slate-800/60 border-2 border-indigo-500/50 rounded-xl space-y-2">
                    <div class="flex justify-between items-center">
                        <span class="font-bold text-indigo-300 text-xs">📊 Treino (Train)</span>
                        <span class="text-[11px] bg-indigo-500/20 text-indigo-300 font-mono px-2 py-0.5 rounded" id="cnt-train-videos">0 Vídeos</span>
                    </div>
                    <div class="flex justify-around pt-2 border-t border-slate-700/50 text-center">
                        <div>
                            <div class="text-[11px] text-slate-400">Frames</div>
                            <div class="text-lg font-bold text-indigo-200 font-mono" id="cnt-train-frames">0</div>
                        </div>
                        <div>
                            <div class="text-[11px] text-slate-400">Caixas</div>
                            <div class="text-lg font-bold text-emerald-300 font-mono" id="cnt-train-boxes">0</div>
                        </div>
                    </div>
                </div>

                <div class="p-4 bg-slate-800/60 border-2 border-purple-500/50 rounded-xl space-y-2">
                    <div class="flex justify-between items-center">
                        <span class="font-bold text-purple-300 text-xs">📊 Validação (Val)</span>
                        <span class="text-[11px] bg-purple-500/20 text-purple-300 font-mono px-2 py-0.5 rounded" id="cnt-val-videos">0 Vídeos</span>
                    </div>
                    <div class="flex justify-around pt-2 border-t border-slate-700/50 text-center">
                        <div>
                            <div class="text-[11px] text-slate-400">Frames</div>
                            <div class="text-lg font-bold text-purple-200 font-mono" id="cnt-val-frames">0</div>
                        </div>
                        <div>
                            <div class="text-[11px] text-slate-400">Caixas</div>
                            <div class="text-lg font-bold text-amber-300 font-mono" id="cnt-val-boxes">0</div>
                        </div>
                    </div>
                </div>

                <div class="p-4 bg-slate-800/60 border-2 border-amber-500/50 rounded-xl space-y-2">
                    <div class="flex justify-between items-center">
                        <span class="font-bold text-amber-300 text-xs">📊 Teste Normal</span>
                        <span class="text-[11px] bg-amber-500/20 text-amber-300 font-mono px-2 py-0.5 rounded" id="cnt-test-normal-videos">0 Vídeos</span>
                    </div>
                    <div class="flex justify-around pt-2 border-t border-slate-700/50 text-center">
                        <div>
                            <div class="text-[11px] text-slate-400">Frames</div>
                            <div class="text-lg font-bold text-amber-200 font-mono" id="cnt-test-normal-frames">0</div>
                        </div>
                        <div>
                            <div class="text-[11px] text-slate-400">Caixas</div>
                            <div class="text-lg font-bold text-amber-400 font-mono" id="cnt-test-normal-boxes">0</div>
                        </div>
                    </div>
                </div>

                <div class="p-4 bg-slate-800/60 border-2 border-rose-500/50 rounded-xl space-y-2">
                    <div class="flex justify-between items-center">
                        <span class="font-bold text-rose-300 text-xs">📊 Teste Estresse</span>
                        <span class="text-[11px] bg-rose-500/20 text-rose-300 font-mono px-2 py-0.5 rounded" id="cnt-test-stress-videos">0 Vídeos</span>
                    </div>
                    <div class="flex justify-around pt-2 border-t border-slate-700/50 text-center">
                        <div>
                            <div class="text-[11px] text-slate-400">Frames</div>
                            <div class="text-lg font-bold text-rose-200 font-mono" id="cnt-test-stress-frames">0</div>
                        </div>
                        <div>
                            <div class="text-[11px] text-slate-400">Caixas</div>
                            <div class="text-lg font-bold text-rose-400 font-mono" id="cnt-test-stress-boxes">0</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Tabela de Atribuição por Unidade Experimental -->
            <div class="space-y-3">
                <h3 class="text-sm font-bold text-slate-300">Defina o papel de cada unidade experimental:</h3>
                <div class="space-y-2" id="video-assignment-list">
                    Carregando vídeos...
                </div>
            </div>

            <div class="pt-3 border-t border-slate-800 flex justify-end">
                <button onclick="materializeRelease()" id="btn-build-release" class="px-5 py-2.5 bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-xs rounded-lg shadow-lg shadow-emerald-600/30 transition">
                    🔨 Materializar Dataset
                </button>
            </div>
        </div>

        <!-- Seção 2: Treinamento do Modelo -->
        <div id="training-section-container" class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6 opacity-60 pointer-events-none transition-all">
            <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
                <h2 class="text-xl font-bold text-amber-400 flex items-center gap-2">
                    <span>⚡</span> Treinamento do Modelo YOLO
                </h2>
                <div id="training-lock-notice" class="text-xs font-semibold text-amber-400/90 bg-amber-500/10 border border-amber-500/30 px-3 py-1 rounded-lg flex items-center gap-1.5">
                    <span>🔒</span> <span>Materialize o dataset acima para liberar o treinamento</span>
                </div>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div>
                    <label class="block text-slate-300 font-medium mb-1">Modelo de Partida</label>
                    <select id="train-model-select" disabled class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2.5 text-white text-xs disabled:opacity-50">
                        <option value="yolo26n.pt">yolo26n.pt (Novo Modelo Base)</option>
                        <option value="yolov8n.pt">yolov8n.pt (Modelo YOLOv8 Nano)</option>
                    </select>
                </div>

                <div>
                    <label class="block text-slate-300 font-medium mb-1">Épocas (epochs)</label>
                    <input type="number" id="train-epochs-input" value="50" disabled class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2.5 text-white disabled:opacity-50">
                </div>

                <div>
                    <label class="block text-slate-300 font-medium mb-1">Tamanho Imagem (imgsz)</label>
                    <input type="number" id="train-imgsz-input" value="640" disabled class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2.5 text-white disabled:opacity-50">
                </div>
            </div>

            <div>
                <button onclick="startTrainingProcess()" id="btn-start-train" disabled class="px-6 py-3 bg-slate-700 text-slate-400 font-bold text-xs rounded-xl transition cursor-not-allowed">
                    🔒 Treinamento Indisponível (Materialize o Dataset Primeiro)
                </button>
            </div>

            <!-- Fila e Treinamentos em Andamento / Agendados -->
            <div id="training-queue-container" class="space-y-3 pt-4 border-t border-slate-800">
                <div class="flex justify-between items-center">
                    <h3 class="font-bold text-slate-200 text-sm flex items-center gap-2">
                        <span>📋</span> Fila e Histórico de Treinamentos para esta Release
                    </h3>
                    <button onclick="loadReleaseJobs()" class="text-xs text-amber-400 hover:underline">🔄 Atualizar Fila</button>
                </div>
                <div id="training-jobs-list" class="space-y-2 text-xs">
                    <p class="text-slate-500 italic">Nenhum treinamento agendado ou em execução.</p>
                </div>
            </div>

            <!-- Terminal de Logs em Tempo Real -->
            <div id="terminal-section" class="hidden space-y-3 pt-4 border-t border-slate-800">
                <div class="flex justify-between items-center">
                    <h3 class="font-bold text-slate-200 text-sm flex items-center gap-2">
                        <span id="active-job-indicator" class="w-2.5 h-2.5 rounded-full bg-amber-400 animate-pulse"></span>
                        Monitoramento do Treinamento: <span id="active-job-id" class="font-mono text-amber-300">...</span>
                    </h3>
                    <div class="flex items-center gap-2">
                        <span id="job-status-tag" class="text-xs font-mono text-amber-300 bg-amber-500/20 px-2.5 py-1 rounded-md border border-amber-500/30">Executando...</span>
                        <button id="btn-stop-active-job" onclick="stopActiveJob()" class="px-3 py-1 bg-rose-600 hover:bg-rose-500 text-white font-bold text-xs rounded-md shadow-md transition">
                            🛑 Parar Treinamento
                        </button>
                    </div>
                </div>
                <pre id="terminal-logs" class="p-4 bg-slate-950 rounded-xl border border-slate-800 text-xs font-mono text-slate-300 h-64 overflow-y-auto">Iniciando monitoramento de logs...</pre>
            </div>
        </div>

    </div>

    <script>
        const urlParams = new URLSearchParams(window.location.search);
        const releaseId = urlParams.get('id') || 'release_default';
        let campaignId = urlParams.get('campaign') || urlParams.get('source');
        let annotationRevisionId = urlParams.get('revision_id') || urlParams.get('revision');

        // Se campaignId não foi passado explicitamente, extrai do releaseId (ex: release_canaleta_pvc_260717_export)
        if (!campaignId && releaseId && releaseId.startsWith('release_')) {
            const parts = releaseId.replace(/^release_/, '').split('_');
            if (parts.length >= 2) {
                // Tenta recompor o ID da campanha
                campaignId = parts.slice(0, -1).join('_');
            }
        }

        let videoList = [];
        let videoAssignments = {};

        async function updateSplitPreview() {
            if (!campaignId) return;
            try {
                const res = await fetch('/api/releases/preview-split', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        campaign_id: campaignId,
                        revision_id: annotationRevisionId,
                        evaluation_level: document.getElementById('evaluation-level').value,
                        assignments: {
                            train: videoList.filter(v => videoAssignments[v] === 'train').map(v => campaignId + '/' + v),
                            val: videoList.filter(v => videoAssignments[v] === 'val').map(v => campaignId + '/' + v),
                            test_normal: videoList.filter(v => videoAssignments[v] === 'test_normal').map(v => campaignId + '/' + v),
                            test_stress: videoList.filter(v => videoAssignments[v] === 'test_stress').map(v => campaignId + '/' + v)
                        }
                    })
                });
                const data = await res.json();
                
                document.getElementById('cnt-train-videos').innerText = `${data.train ? data.train.videos : 0} Unidade(s)`;
                document.getElementById('cnt-train-frames').innerText = data.train ? data.train.frames : 0;
                document.getElementById('cnt-train-boxes').innerText = data.train ? data.train.boxes : 0;

                document.getElementById('cnt-val-videos').innerText = `${data.val ? data.val.videos : 0} Unidade(s)`;
                document.getElementById('cnt-val-frames').innerText = data.val ? data.val.frames : 0;
                document.getElementById('cnt-val-boxes').innerText = data.val ? data.val.boxes : 0;

                document.getElementById('cnt-test-normal-videos').innerText = `${data.test_normal ? data.test_normal.videos : 0} Unidade(s)`;
                document.getElementById('cnt-test-normal-frames').innerText = data.test_normal ? data.test_normal.frames : 0;
                document.getElementById('cnt-test-normal-boxes').innerText = data.test_normal ? data.test_normal.boxes : 0;

                document.getElementById('cnt-test-stress-videos').innerText = `${data.test_stress ? data.test_stress.videos : 0} Unidade(s)`;
                document.getElementById('cnt-test-stress-frames').innerText = data.test_stress ? data.test_stress.frames : 0;
                document.getElementById('cnt-test-stress-boxes').innerText = data.test_stress ? data.test_stress.boxes : 0;
                const quality = data.quality_assessment || {};
                const qualityBox = document.getElementById('split-quality-assessment');
                const blockers = quality.blocking || [];
                const warnings = quality.warnings || [];
                if (blockers.length > 0) {
                    qualityBox.className = 'p-3 bg-rose-500/10 border border-rose-500/30 rounded-xl text-xs text-rose-300';
                    qualityBox.innerHTML = `<strong>Materialização bloqueada:</strong><br>${blockers.map(escapeHtml).join('<br>')}`;
                } else if (warnings.length > 0) {
                    qualityBox.className = 'p-3 bg-amber-500/10 border border-amber-500/30 rounded-xl text-xs text-amber-200';
                    qualityBox.innerHTML = `<strong>Avisos de suficiência:</strong><br>${warnings.map(escapeHtml).join('<br>')}`;
                } else {
                    qualityBox.className = 'p-3 bg-emerald-500/10 border border-emerald-500/30 rounded-xl text-xs text-emerald-300';
                    qualityBox.innerText = 'Os requisitos estruturais mínimos foram atendidos.';
                }
            } catch (err) {
                console.error(err);
            }
        }

        function setVideoRole(vName, role) {
            videoAssignments[vName] = role;
            updateSplitPreview();
        }

        async function materializeRelease() {
            try {
                const targetRelId = document.getElementById('input-release-id').value.trim() || releaseId;
                const trainV = videoList.filter(v => videoAssignments[v] === 'train').map(v => campaignId + '/' + v);
                const valV = videoList.filter(v => videoAssignments[v] === 'val').map(v => campaignId + '/' + v);
                const testNormalV = videoList.filter(v => videoAssignments[v] === 'test_normal').map(v => campaignId + '/' + v);
                const testStressV = videoList.filter(v => videoAssignments[v] === 'test_stress').map(v => campaignId + '/' + v);

                // Criar release
                const resC = await fetch('/api/releases', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        release_id: targetRelId,
                        campaigns: [campaignId],
                        evaluation_level: document.getElementById('evaluation-level').value,
                        annotation_revisions: annotationRevisionId ? { [campaignId]: annotationRevisionId } : {},
                        assignments: {
                            train: trainV,
                            val: valV,
                            test_normal: testNormalV,
                            test_stress: testStressV
                        }
                    })
                });
                const dataC = await resC.json();
                if (!resC.ok) throw new Error(dataC.detail || 'Erro ao registrar release.');

                // Materializar
                const resB = await fetch(`/api/releases/${targetRelId}/build`, { method: 'POST' });
                const dataB = await resB.json();
                if (!resB.ok) throw new Error(dataB.detail || 'Erro na materialização');

                alert('Dataset materializado com sucesso!');
                document.getElementById('rel-status-badge').innerText = 'Status: Materializada';
                document.getElementById('rel-title').innerText = targetRelId;

                // Desbloquear seção de treinamento
                isReleaseMaterialized = true;
                enableTrainingSection();
            } catch (err) {
                alert(err.message);
            }
        }

        let activeJobId = null;
        let jobPollInterval = null;

        async function loadReleaseJobs() {
            try {
                const res = await fetch('/api/jobs');
                const jobs = await res.json();
                const targetRelId = document.getElementById('input-release-id')?.value.trim() || releaseId;
                const releaseJobs = jobs.filter(j => j.target === targetRelId || j.target === releaseId);
                const listDiv = document.getElementById('training-jobs-list');

                if (!releaseJobs || releaseJobs.length === 0) {
                    listDiv.innerHTML = '<p class="text-slate-500 italic">Nenhum treinamento agendado ou em execução.</p>';
                    return;
                }

                let html = '';
                releaseJobs.forEach(job => {
                    let statusBadge = '';
                    let actionBtn = '';

                    if (job.status === 'queued') {
                        statusBadge = '<span class="px-2 py-0.5 bg-amber-500/10 text-amber-300 border border-amber-500/20 rounded text-[11px] font-semibold">⏳ Na Fila</span>';
                        actionBtn = `<button onclick="cancelQueuedJob('${job.id}')" class="px-2.5 py-1 bg-slate-800 hover:bg-rose-900/40 text-rose-300 border border-slate-700 hover:border-rose-500/50 rounded text-xs transition">❌ Remover da Fila</button>`;
                    } else if (job.status === 'running') {
                        statusBadge = '<span class="px-2 py-0.5 bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 rounded text-[11px] font-semibold animate-pulse">⚡ Em Execução</span>';
                        actionBtn = `<button onclick="stopActiveJob('${job.id}')" class="px-2.5 py-1 bg-rose-600/20 text-rose-300 border border-rose-500/30 rounded text-xs hover:bg-rose-600 hover:text-white transition">🛑 Parar Treino</button>`;
                    } else if (job.status === 'completed') {
                        statusBadge = '<span class="px-2 py-0.5 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded text-[11px] font-semibold">✅ Concluído</span>';
                    } else if (job.status === 'stopped' || job.status === 'cancelled') {
                        statusBadge = `<span class="px-2 py-0.5 bg-slate-800 text-slate-400 border border-slate-700 rounded text-[11px] font-semibold">⏹ Interrompido (${job.status})</span>`;
                    } else {
                        statusBadge = `<span class="px-2 py-0.5 bg-rose-500/10 text-rose-400 border border-rose-500/20 rounded text-[11px] font-semibold">❌ ${job.status}</span>`;
                    }

                    const meta = job.metadata || {};
                    const metaStr = meta.model ? `Modelo: ${meta.model} | Épocas: ${meta.epochs} | imgsz: ${meta.imgsz}` : '';

                    html += `
                        <div class="p-3 bg-slate-800/40 border border-slate-800 rounded-lg flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                            <div class="space-y-0.5">
                                <div class="flex items-center gap-2">
                                    <span class="font-mono font-bold text-slate-200">${job.id}</span>
                                    ${statusBadge}
                                </div>
                                ${metaStr ? `<div class="text-[11px] text-slate-400 font-sans">${metaStr}</div>` : ''}
                            </div>
                            <div class="flex items-center gap-2">
                                <button onclick="viewJobLogs('${job.id}')" class="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 rounded text-xs transition">👁 Ver Logs</button>
                                ${actionBtn}
                            </div>
                        </div>
                    `;
                });
                listDiv.innerHTML = html;

                // Se houver um job rodando, ajusta o polling automático para ele
                const runningJob = releaseJobs.find(j => j.status === 'running');
                if (runningJob) {
                    if (activeJobId !== runningJob.id) {
                        viewJobLogs(runningJob.id);
                    }
                }

            } catch (err) {
                console.error(err);
            }
        }

        async function cancelQueuedJob(jobId) {
            try {
                const res = await fetch(`/api/jobs/${jobId}/cancel`, { method: 'POST' });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || 'Erro ao cancelar o treinamento agendado.');
                loadReleaseJobs();
            } catch (err) {
                alert(err.message);
            }
        }

        async function stopActiveJob(targetJobId) {
            const jId = targetJobId || activeJobId;
            if (!jId) return;
            if (!confirm('Deseja realmente interromper este treinamento em andamento?')) return;

            try {
                const res = await fetch(`/api/jobs/${jId}/stop`, { method: 'POST' });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || 'Erro ao parar o treinamento.');
                loadReleaseJobs();
            } catch (err) {
                alert(err.message);
            }
        }

        function viewJobLogs(jobId) {
            activeJobId = jobId;
            document.getElementById('active-job-id').innerText = jobId;
            document.getElementById('terminal-section').classList.remove('hidden');
            if (jobPollInterval) clearInterval(jobPollInterval);
            pollJob(jobId);
        }

        async function startTrainingProcess() {
            const targetRelId = document.getElementById('input-release-id')?.value.trim() || releaseId;
            const model = document.getElementById('train-model-select').value;
            const epochs = parseInt(document.getElementById('train-epochs-input').value);
            const imgsz = parseInt(document.getElementById('train-imgsz-input').value);

            try {
                const res = await fetch(`/api/releases/${targetRelId}/start-train`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: model,
                        epochs: epochs,
                        imgsz: imgsz
                    })
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || 'Erro ao agendar treinamento');

                loadReleaseJobs();
                viewJobLogs(data.id);
            } catch (err) {
                alert(err.message);
            }
        }

        async function pollJob(jobId) {
            const logBox = document.getElementById('terminal-logs');
            const statusTag = document.getElementById('job-status-tag');

            const fetchStatus = async () => {
                try {
                    const res = await fetch(`/api/jobs/${jobId}`);
                    const job = await res.json();
                    
                    statusTag.innerText = `Status: ${job.status}`;
                    logBox.innerText = job.log || 'Aguardando logs...';
                    logBox.scrollTop = logBox.scrollHeight;

                    if (job.status === 'completed' || job.status === 'failed' || job.status === 'stopped' || job.status === 'cancelled') {
                        if (jobPollInterval) {
                            clearInterval(jobPollInterval);
                            jobPollInterval = null;
                        }
                        loadReleaseJobs();
                    }
                } catch (err) {
                    console.error(err);
                }
            };

            fetchStatus();
            if (jobPollInterval) clearInterval(jobPollInterval);
            jobPollInterval = setInterval(fetchStatus, 2000);
        }

        let isReleaseMaterialized = false;

        function lockReleaseConfiguration() {
            const inputRelId = document.getElementById('input-release-id');
            const btnBuild = document.getElementById('btn-build-release');
            if (inputRelId) inputRelId.disabled = true;
            if (btnBuild) {
                btnBuild.disabled = true;
                btnBuild.className = 'px-5 py-2.5 bg-slate-800 text-slate-400 font-medium text-xs rounded-lg transition cursor-not-allowed border border-slate-700';
                btnBuild.innerText = '✅ Dataset Materializado (Imutável)';
            }
            document.querySelectorAll('input[name^="role_"]').forEach(radio => {
                radio.disabled = true;
            });
        }

        function enableTrainingSection() {
            const container = document.getElementById('training-section-container');
            const notice = document.getElementById('training-lock-notice');
            const btn = document.getElementById('btn-start-train');
            const selModel = document.getElementById('train-model-select');
            const inEpochs = document.getElementById('train-epochs-input');
            const inImgsz = document.getElementById('train-imgsz-input');

            if (container) {
                container.classList.remove('opacity-60', 'pointer-events-none');
            }
            if (notice) {
                notice.className = 'text-xs font-semibold text-emerald-400/90 bg-emerald-500/10 border border-emerald-500/30 px-3 py-1 rounded-lg flex items-center gap-1.5';
                notice.innerHTML = '<span>✅</span> <span>Dataset materializado e pronto para treino</span>';
            }
            if (selModel) selModel.disabled = false;
            if (inEpochs) inEpochs.disabled = false;
            if (inImgsz) inImgsz.disabled = false;
            if (btn) {
                btn.disabled = false;
                btn.className = 'px-6 py-3 bg-amber-600 hover:bg-amber-500 text-white font-bold text-xs rounded-xl shadow-lg shadow-amber-600/30 transition cursor-pointer';
                btn.innerHTML = '🚀 Iniciar Treinamento';
            }
            lockReleaseConfiguration();
        }

        function disableTrainingSection() {
            const container = document.getElementById('training-section-container');
            const notice = document.getElementById('training-lock-notice');
            const btn = document.getElementById('btn-start-train');
            const selModel = document.getElementById('train-model-select');
            const inEpochs = document.getElementById('train-epochs-input');
            const inImgsz = document.getElementById('train-imgsz-input');

            if (container) {
                container.classList.add('opacity-60', 'pointer-events-none');
            }
            if (notice) {
                notice.className = 'text-xs font-semibold text-amber-400/90 bg-amber-500/10 border border-amber-500/30 px-3 py-1 rounded-lg flex items-center gap-1.5';
                notice.innerHTML = '<span>🔒</span> <span>Materialize o dataset acima para liberar o treinamento</span>';
            }
            if (selModel) selModel.disabled = true;
            if (inEpochs) inEpochs.disabled = true;
            if (inImgsz) inImgsz.disabled = true;
            if (btn) {
                btn.disabled = true;
                btn.className = 'px-6 py-3 bg-slate-700 text-slate-400 font-bold text-xs rounded-xl transition cursor-not-allowed';
                btn.innerHTML = '🔒 Treinamento Indisponível (Materialize o Dataset Primeiro)';
            }
        }

        async function initReleasePage() {
            document.getElementById('rel-title').innerText = releaseId || 'Release';
            
            try {
                // Tenta consultar as informações da release via API se ela já existir
                let existingReleaseInfo = null;
                if (releaseId) {
                    try {
                        const resV = await fetch(`/api/releases/${releaseId}`);
                        if (resV.ok) {
                            existingReleaseInfo = await resV.json();
                        }
                    } catch (e) {
                        console.warn('Release ainda não criada:', e);
                    }
                }

                // Se a release existir, extrai a campanha/origem vinculada a ela no arquivo de configuração
                if (existingReleaseInfo && (existingReleaseInfo.sources || existingReleaseInfo.campaigns)) {
                    const srcList = existingReleaseInfo.sources || existingReleaseInfo.campaigns;
                    if (srcList.length > 0) {
                        campaignId = srcList[0];
                    }
                    const storedRevisions = existingReleaseInfo.annotation_revisions || {};
                    if (storedRevisions[campaignId]) {
                        annotationRevisionId = storedRevisions[campaignId];
                    }
                }

                // Se a URL não tiver campaignId nem conseguir derivar, busca a primeira campanha disponível
                if (!campaignId) {
                    const resC = await fetch('/api/campaigns');
                    const campaigns = await resC.json();
                    if (campaigns && campaigns.length > 0) {
                        campaignId = campaigns[0];
                    }
                }

                const inputRelId = document.getElementById('input-release-id');
                if (inputRelId) inputRelId.value = releaseId || (campaignId ? `release_${campaignId}` : 'release_01');

                // Verificar se a release já foi materializada previamente (manifesto ou flag materialized)
                if (existingReleaseInfo && (existingReleaseInfo.materialized || existingReleaseInfo.build_report)) {
                    isReleaseMaterialized = true;
                    document.getElementById('rel-status-badge').innerText = 'Status: Materializada';
                    enableTrainingSection();
                } else {
                    disableTrainingSection();
                }

                if (!campaignId) {
                    document.getElementById('video-assignment-list').innerHTML = '<p class="text-rose-400 text-xs py-2">Identificador da origem/campanha não encontrado na URL.</p>';
                    return;
                }

                const resSt = await fetch(`/api/campaigns/${campaignId}`);
                if (!resSt.ok) {
                    throw new Error(`Falha ao carregar dados da origem: ${resSt.statusText}`);
                }
                const st = await resSt.json();
                if (!annotationRevisionId) {
                    annotationRevisionId = st.latest_annotation_revision || null;
                }
                const revision = (st.annotation_revisions || []).find(
                    item => item.revision_id === annotationRevisionId
                );
                if (!annotationRevisionId || !revision) {
                    throw new Error('A revisão de anotação selecionada não existe nesta origem.');
                }
                const report = revision;
                document.getElementById('release-revision-id').innerText = annotationRevisionId;
                const videoDetails = st.video_details || [];
                const notesMap = {};
                videoDetails.forEach(v => {
                    notesMap[v.name] = v.note || '';
                });
                (st.capture_units || []).forEach(unit => {
                    notesMap[unit.unit_id] = unit.note || unit.condition || '';
                });

                videoList = Object.keys(report.per_unit || report.per_video || {});
                if (videoList.length === 0 && (st.capture_units || []).length > 0) {
                    videoList = st.capture_units.map(unit => unit.unit_id);
                }

                if (videoList.length === 0) {
                    document.getElementById('video-assignment-list').innerHTML = '<p class="text-amber-400 text-xs py-2">Nenhum vídeo encontrado para esta origem.</p>';
                    return;
                }

                // Atribuir por padrão com base na nota da unidade.
                // Para releases existentes, preservar exatamente a configuração registrada.
                const storedAssignments = existingReleaseInfo ? (existingReleaseInfo.assignments || {}) : {};
                if (existingReleaseInfo && existingReleaseInfo.evaluation_level) {
                    document.getElementById('evaluation-level').value =
                        existingReleaseInfo.evaluation_level;
                }
                const storedRoleByVideo = {};
                Object.entries(storedAssignments).forEach(([role, items]) => {
                    (items || []).forEach(item => {
                        const prefix = `${campaignId}/`;
                        const videoName = item.startsWith(prefix) ? item.slice(prefix.length) : item;
                        storedRoleByVideo[videoName] = role;
                    });
                });
                videoList.forEach((v, idx) => {
                    if (storedRoleByVideo[v]) {
                        videoAssignments[v] = storedRoleByVideo[v];
                        return;
                    }
                    const noteStr = (notesMap[v] || '').toLowerCase().trim();
                    if (noteStr.includes('estresse') || noteStr.includes('stress')) {
                        videoAssignments[v] = 'test_stress';
                    } else if (noteStr.includes('normal')) {
                        videoAssignments[v] = 'test_normal';
                    } else if (noteStr.includes('validação') || noteStr.includes('validacao') || noteStr.includes('val')) {
                        videoAssignments[v] = 'val';
                    } else if (noteStr.includes('treino') || noteStr.includes('train')) {
                        videoAssignments[v] = 'train';
                    } else if (videoList.length >= 3 && idx === videoList.length - 1) {
                        videoAssignments[v] = 'test_normal';
                    } else if (videoList.length >= 2 && idx === videoList.length - 2) {
                        videoAssignments[v] = 'val';
                    } else {
                        videoAssignments[v] = 'train';
                    }
                });
                if (videoList.length === 1 && !notesMap[videoList[0]]) videoAssignments[videoList[0]] = 'train';

                // Renderizar tabela
                let html = '';
                videoList.forEach(v => {
                    const role = videoAssignments[v];
                    const note = notesMap[v] || '';
                    html += `
                        <div class="p-3.5 bg-slate-800/40 border border-slate-800 rounded-lg flex flex-col md:flex-row md:items-center justify-between gap-3 text-xs">
                            <div class="space-y-1">
                                <div class="font-mono text-slate-200 font-bold">${escapeHtml(v)}</div>
                                <div class="text-[10px] text-slate-500">Unidade experimental</div>
                                ${note ? `<div class="text-[11px] text-amber-300/90 flex items-center gap-1 font-sans"><span>📝</span> <span>${escapeHtml(note)}</span></div>` : '<div class="text-[11px] text-slate-500 font-sans italic">Sem observações</div>'}
                            </div>
                            <div class="flex flex-wrap gap-2.5">
                                <label class="flex items-center gap-1 cursor-pointer bg-slate-800 px-2.5 py-1 rounded border border-slate-700 hover:border-indigo-500/50">
                                    <input type="radio" name="role_${v}" value="train" ${role==='train'?'checked':''} onchange="setVideoRole('${v}', 'train')" class="text-indigo-600">
                                    <span class="text-indigo-300 font-medium">Treino (Train)</span>
                                </label>
                                <label class="flex items-center gap-1 cursor-pointer bg-slate-800 px-2.5 py-1 rounded border border-slate-700 hover:border-purple-500/50">
                                    <input type="radio" name="role_${v}" value="val" ${role==='val'?'checked':''} onchange="setVideoRole('${v}', 'val')" class="text-purple-600">
                                    <span class="text-purple-300 font-medium">Validação (Val)</span>
                                </label>
                                <label class="flex items-center gap-1 cursor-pointer bg-slate-800 px-2.5 py-1 rounded border border-slate-700 hover:border-amber-500/50">
                                    <input type="radio" name="role_${v}" value="test_normal" ${role==='test_normal'?'checked':''} onchange="setVideoRole('${v}', 'test_normal')" class="text-amber-600">
                                    <span class="text-amber-300 font-medium">Teste Normal</span>
                                </label>
                                <label class="flex items-center gap-1 cursor-pointer bg-slate-800 px-2.5 py-1 rounded border border-slate-700 hover:border-rose-500/50">
                                    <input type="radio" name="role_${v}" value="test_stress" ${role==='test_stress'?'checked':''} onchange="setVideoRole('${v}', 'test_stress')" class="text-rose-600">
                                    <span class="text-rose-300 font-medium">Teste Estresse</span>
                                </label>
                            </div>
                        </div>
                    `;
                });
                document.getElementById('video-assignment-list').innerHTML = html;
                updateSplitPreview();

                if (isReleaseMaterialized) {
                    lockReleaseConfiguration();
                }

                // Carregar Modelos em models/
                const resM = await fetch('/api/models');
                const models = await resM.json();
                const sel = document.getElementById('train-model-select');
                if (models.length > 0) {
                    models.forEach(m => {
                        const opt = document.createElement('option');
                        opt.value = m;
                        opt.innerText = `${m} (Modelo em models/)`;
                        sel.appendChild(opt);
                    });
                }

                // Carregar Fila de Treinamentos
                loadReleaseJobs();

            } catch (err) {
                console.error(err);
                document.getElementById('video-assignment-list').innerHTML = `<p class="text-rose-400 text-xs py-2">Erro ao carregar vídeos: ${escapeHtml(err.message)}</p>`;
            }
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        initReleasePage();
    </script>
</body>
</html>"""

    @app.get("/training.html", response_class=HTMLResponse)
    def training_detail_page():
        return """<!DOCTYPE html>
<html lang="pt-BR" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Studio - Detalhes do Treinamento</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background-color: #0f172a; color: #f8fafc; font-family: system-ui, -apple-system, sans-serif; }
    </style>
</head>
<body class="p-6 md:p-10">
    <div class="max-w-6xl mx-auto space-y-8">
        <!-- Header -->
        <header class="flex flex-col md:flex-row justify-between items-start md:items-center pb-6 border-b border-slate-800 gap-4">
            <div>
                <div class="flex items-center gap-3">
                    <a href="/" class="text-xs text-indigo-400 font-semibold hover:underline">&larr; Voltar para a Início</a>
                    <span id="train-status-badge" class="px-2.5 py-0.5 bg-amber-500/10 text-amber-400 border border-amber-500/20 rounded-md text-xs font-semibold">Carregando...</span>
                </div>
                <h1 id="train-title" class="text-3xl font-extrabold tracking-tight text-amber-400 mt-2">Treinamento</h1>
                <p class="text-slate-400 text-sm mt-1">Métricas de treinamento, artefatos gerados e promoção de modelo</p>
            </div>

            <!-- Botão de Ação: Promover Modelo -->
            <div id="promote-action-container" class="hidden flex items-center gap-2">
                <button onclick="openPromoteModal()" class="px-5 py-2.5 bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs rounded-xl shadow-lg shadow-emerald-600/30 transition flex items-center gap-2">
                    <span>🏆</span> <span>Promover para pasta models/</span>
                </button>
            </div>
        </header>

        <!-- Módulos de Métricas Quantitativas e Identificação do Modelo -->
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-1 shadow-lg">
                <div class="text-slate-400 font-medium">mAP50 (Acurácia 50%)</div>
                <div id="stat-map50" class="text-2xl font-extrabold text-emerald-400 font-mono">--</div>
                <div class="text-[11px] text-slate-500">Validação durante o treinamento</div>
            </div>
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-1 shadow-lg">
                <div class="text-slate-400 font-medium">mAP50-95</div>
                <div id="stat-map5095" class="text-2xl font-extrabold text-indigo-400 font-mono">--</div>
                <div class="text-[11px] text-slate-500">Validação durante o treinamento</div>
            </div>
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-1 shadow-lg">
                <div class="text-slate-400 font-medium">Precisão / Precisão (P)</div>
                <div id="stat-precision" class="text-2xl font-extrabold text-amber-400 font-mono">--</div>
                <div class="text-[11px] text-slate-500">Taxa de falsos positivos</div>
            </div>
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 space-y-1 shadow-lg">
                <div class="text-slate-400 font-medium">Revocação / Recall (R)</div>
                <div id="stat-recall" class="text-2xl font-extrabold text-purple-400 font-mono">--</div>
                <div class="text-[11px] text-slate-500">Taxa de detecção completa</div>
            </div>
        </div>

        <section class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
            <div>
                <h2 class="text-lg font-bold text-slate-200">🧪 Avaliação final e robustez</h2>
                <p class="text-xs text-slate-400 mt-1">
                    O mesmo <code class="text-emerald-300">best.pt</code> é avaliado após o treino.
                    O teste de estresse nunca participa da seleção do modelo.
                </p>
            </div>
            <div class="overflow-x-auto">
                <table class="w-full text-xs">
                    <thead class="text-slate-400 border-b border-slate-800">
                        <tr>
                            <th class="text-left py-2">Métrica</th>
                            <th class="text-right py-2">Teste normal</th>
                            <th class="text-right py-2">Teste de estresse</th>
                            <th class="text-right py-2">Queda</th>
                        </tr>
                    </thead>
                    <tbody id="evaluation-metrics-body" class="font-mono text-slate-200"></tbody>
                </table>
            </div>
            <div id="evaluation-status" class="text-xs text-slate-500">Carregando avaliações...</div>
        </section>

        <!-- Grade de Detalhes do Modelo & Arquivos -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <!-- Coluna Principal (2 Cols): Gráficos e Imagens -->
            <div class="lg:col-span-2 space-y-6">
                <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
                    <h2 class="text-lg font-bold text-slate-200 flex items-center gap-2">
                        <span>📊</span> Métricas e Gráficos de Resultados
                    </h2>
                    <div id="train-images-container" class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <p class="text-slate-500 text-xs italic">Carregando visualizações...</p>
                    </div>
                </div>

                <!-- Terminal / Log do Treinamento -->
                <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-3">
                    <h2 class="text-lg font-bold text-slate-200 flex items-center gap-2">
                        <span>📄</span> Log de Saída do Treinamento
                    </h2>
                    <pre id="train-logs-box" class="p-4 bg-slate-950 rounded-xl border border-slate-800 text-xs font-mono text-slate-300 h-80 overflow-y-auto">Carregando logs...</pre>
                </div>
            </div>

            <!-- Coluna Lateral: Especificações do Modelo e Dataset -->
            <div class="space-y-6">
                <!-- Informações do Modelo & Dataset -->
                <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
                    <h2 class="text-lg font-bold text-slate-200 flex items-center gap-2">
                        <span>ℹ️</span> Identificação e Origem
                    </h2>
                    <div class="space-y-3 text-xs">
                        <div>
                            <div class="text-slate-400 font-medium">Modelo Base Inicial</div>
                            <div id="info-base-model" class="font-mono font-bold text-amber-300 mt-0.5">--</div>
                        </div>
                        <div>
                            <div class="text-slate-400 font-medium">Dataset / Release Materializado</div>
                            <div id="info-release-id" class="font-mono font-bold text-emerald-400 mt-0.5">--</div>
                        </div>
                        <div>
                            <div class="text-slate-400 font-medium">Nível de Avaliação</div>
                            <div id="info-evaluation-level" class="font-mono font-bold text-amber-300 mt-0.5">--</div>
                        </div>
                        <div>
                            <div class="text-slate-400 font-medium">Origens de Dados</div>
                            <div id="info-sources" class="font-mono text-slate-200 mt-0.5">--</div>
                        </div>
                        <div class="grid grid-cols-2 gap-2 pt-1 border-t border-slate-800">
                            <div>
                                <div class="text-slate-400 font-medium">Épocas Treinadas</div>
                                <div id="info-epochs" class="font-mono font-bold text-slate-200 mt-0.5">--</div>
                            </div>
                            <div>
                                <div class="text-slate-400 font-medium">Resolução (imgsz)</div>
                                <div id="info-imgsz" class="font-mono font-bold text-slate-200 mt-0.5">--</div>
                            </div>
                        </div>
                        <div class="grid grid-cols-2 gap-2 pt-1 border-t border-slate-800">
                            <div>
                                <div class="text-slate-400 font-medium">Total de Imagens</div>
                                <div id="info-total-images" class="font-mono font-bold text-slate-200 mt-0.5">--</div>
                            </div>
                            <div>
                                <div class="text-slate-400 font-medium">Total de Marcações</div>
                                <div id="info-total-boxes" class="font-mono font-bold text-slate-200 mt-0.5">--</div>
                            </div>
                        </div>
                        <div class="pt-1 border-t border-slate-800">
                            <div class="text-slate-400 font-medium">Tempo de Execução</div>
                            <div id="info-duration" class="font-mono font-bold text-indigo-300 mt-0.5">--</div>
                        </div>
                    </div>
                </div>

                <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
                    <h2 class="text-lg font-bold text-slate-200 flex items-center gap-2">
                        <span>📦</span> Artefatos e Pesos (.pt)
                    </h2>
                    <div class="space-y-3 text-xs">
                        <div class="p-3 bg-slate-800/40 border border-slate-800 rounded-xl flex items-center justify-between">
                            <div>
                                <div class="font-bold text-emerald-400">best.pt</div>
                                <div class="text-[11px] text-slate-400">Melhores pesos de validação</div>
                            </div>
                            <span id="badge-best-status" class="px-2 py-0.5 bg-slate-800 text-slate-400 rounded border border-slate-700">Verificando...</span>
                        </div>
                        <div class="p-3 bg-slate-800/40 border border-slate-800 rounded-xl flex items-center justify-between">
                            <div>
                                <div class="font-bold text-indigo-400">last.pt</div>
                                <div class="text-[11px] text-slate-400">Pesos da última época</div>
                            </div>
                            <span id="badge-last-status" class="px-2 py-0.5 bg-slate-800 text-slate-400 rounded border border-slate-700">Verificando...</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Modal de Promoção de Modelo -->
    <div id="promote-modal" class="fixed inset-0 bg-slate-950/80 backdrop-blur-sm hidden items-center justify-center p-4 z-50">
        <div class="bg-slate-900 border border-slate-800 rounded-2xl max-w-md w-full p-6 space-y-5 shadow-2xl">
            <div>
                <h3 class="text-lg font-bold text-emerald-400">🏆 Promover Modelo para models/</h3>
                <p class="text-xs text-slate-400 mt-1">Copie o arquivo <code class="text-emerald-300 font-mono">best.pt</code> deste treinamento para o diretório de modelos do workspace para usá-lo em novas predições.</p>
            </div>

            <div class="space-y-2">
                <label class="block text-xs font-semibold text-slate-300">Nome do Modelo em models/</label>
                <input type="text" id="input-promote-name" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2.5 text-xs text-white focus:outline-none focus:border-emerald-500 font-mono">
            </div>

            <div id="promote-error" class="hidden text-rose-400 text-xs font-medium"></div>

            <div class="flex justify-end gap-3 pt-2">
                <button onclick="closePromoteModal()" class="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 font-semibold text-xs rounded-lg transition">Cancelar</button>
                <button onclick="executePromote()" id="btn-confirm-promote" class="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs rounded-lg transition shadow-lg shadow-emerald-600/30">Promover Modelo</button>
            </div>
        </div>
    </div>

    <script>
        const urlParams = new URLSearchParams(window.location.search);
        const trainingId = urlParams.get('id');

        function formatPercent(value) {
            return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '--';
        }

        function renderEvaluations(evaluations, robustness) {
            const normal = evaluations.test_normal || {};
            const stress = evaluations.test_stress || {};
            const labels = {
                precision: 'Precisão',
                recall: 'Recall',
                map50: 'mAP50',
                map50_95: 'mAP50-95'
            };
            const rows = Object.entries(labels).map(([key, label]) => {
                const drop = (robustness[key] || {}).drop_absolute;
                return `<tr class="border-b border-slate-800/60">
                    <td class="py-2 font-sans">${label}</td>
                    <td class="py-2 text-right">${formatPercent(normal[key])}</td>
                    <td class="py-2 text-right">${formatPercent(stress[key])}</td>
                    <td class="py-2 text-right ${typeof drop === 'number' && drop > 0 ? 'text-rose-400' : 'text-slate-400'}">${formatPercent(drop)}</td>
                </tr>`;
            });
            document.getElementById('evaluation-metrics-body').innerHTML = rows.join('');

            const describe = (label, value) => {
                if (value.status === 'completed') {
                    return `${label}: ${value.images || 0} imagens, ${value.boxes || 0} caixas`;
                }
                if (value.status === 'failed') {
                    return `${label}: falhou (${value.error || 'erro não informado'})`;
                }
                return `${label}: não disponível nesta versão`;
            };
            document.getElementById('evaluation-status').innerText =
                `${describe('Teste normal', normal)} · ${describe('Teste de estresse', stress)}`;
        }

        async function initTrainingPage() {
            if (!trainingId) {
                alert('ID de treinamento não informado!');
                window.location.href = '/';
                return;
            }

            document.getElementById('train-title').innerText = trainingId;
            document.getElementById('input-promote-name').value = `${trainingId}_best.pt`;

            try {
                const res = await fetch(`/api/trainings/${encodeURIComponent(trainingId)}`);
                if (!res.ok) throw new Error('Treinamento não encontrado.');
                const data = await res.json();

                // Status
                const statusBadge = document.getElementById('train-status-badge');
                if (data.status === 'completed') {
                    statusBadge.innerText = 'Status: Concluído';
                    statusBadge.className = 'px-2.5 py-0.5 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded-md text-xs font-semibold';
                } else {
                    statusBadge.innerText = `Status: ${data.status}`;
                }

                // Preenchimento das Métricas Numéricas Quantitativas
                const m = data.metrics || {};
                document.getElementById('stat-map50').innerText = m.map50 !== null ? `${(m.map50 * 100).toFixed(1)}%` : '--';
                document.getElementById('stat-map5095').innerText = m.map50_95 !== null ? `${(m.map50_95 * 100).toFixed(1)}%` : '--';
                document.getElementById('stat-precision').innerText = m.precision !== null ? `${(m.precision * 100).toFixed(1)}%` : '--';
                document.getElementById('stat-recall').innerText = m.recall !== null ? `${(m.recall * 100).toFixed(1)}%` : '--';
                renderEvaluations(data.evaluations || {}, data.robustness || {});

                // Preenchimento dos Dados do Modelo e Release
                const args = data.args || {};
                const rel = data.release || {};
                const buildRep = rel.build_report || {};

                document.getElementById('info-base-model').innerText = args.model || 'yolo26n.pt';
                document.getElementById('info-release-id').innerText = rel.release_id || trainingId;
                document.getElementById('info-evaluation-level').innerText =
                    rel.evaluation_level || 'legacy';
                document.getElementById('info-sources').innerText = (rel.sources || []).join(', ') || 'N/A';
                document.getElementById('info-epochs').innerText = `${m.completed_epochs || 0} / ${args.epochs || '--'}`;
                document.getElementById('info-imgsz').innerText = `${args.imgsz || '--'} px`;

                document.getElementById('info-total-images').innerText = buildRep.images !== undefined ? buildRep.images : '--';
                document.getElementById('info-total-boxes').innerText = buildRep.boxes !== undefined ? buildRep.boxes : '--';

                // Formatação do tempo de execução
                if (data.duration_seconds !== null && data.duration_seconds !== undefined) {
                    const sec = data.duration_seconds;
                    const mins = Math.floor(sec / 60);
                    const secs = sec % 60;
                    document.getElementById('info-duration').innerText = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
                } else {
                    document.getElementById('info-duration').innerText = '--';
                }

                // Pesos
                document.getElementById('badge-best-status').innerText = data.has_best ? 'Disponível' : 'Ausente';
                document.getElementById('badge-best-status').className = data.has_best ? 'px-2 py-0.5 bg-emerald-500/20 text-emerald-400 rounded text-[11px] font-semibold' : 'px-2 py-0.5 bg-slate-800 text-slate-500 rounded text-[11px]';

                document.getElementById('badge-last-status').innerText = data.has_last ? 'Disponível' : 'Ausente';
                document.getElementById('badge-last-status').className = data.has_last ? 'px-2 py-0.5 bg-indigo-500/20 text-indigo-400 rounded text-[11px] font-semibold' : 'px-2 py-0.5 bg-slate-800 text-slate-500 rounded text-[11px]';

                // Liberar botão de promoção se tiver best.pt
                if (data.has_best) {
                    document.getElementById('promote-action-container').classList.remove('hidden');
                }

                // Logs
                const logBox = document.getElementById('train-logs-box');
                logBox.innerText = data.log || 'Nenhum log gravado.';
                logBox.scrollTop = logBox.scrollHeight;

                // Imagens de Gráficos e Validação
                const imgContainer = document.getElementById('train-images-container');
                if (!data.images || data.images.length === 0) {
                    imgContainer.innerHTML = '<p class="text-slate-500 text-xs italic col-span-2">Nenhum gráfico gerado ainda.</p>';
                } else {
                    imgContainer.innerHTML = data.images.map(img => `
                        <div class="bg-slate-950 p-2 border border-slate-800 rounded-xl space-y-1">
                            <div class="text-[11px] font-mono text-slate-400 truncate" title="${escapeHtml(img)}">${escapeHtml(img)}</div>
                            <img src="/runs/detect/${encodeURIComponent(trainingId)}/${encodeURIComponent(img)}" class="w-full h-auto rounded-lg object-contain bg-slate-900 border border-slate-800">
                        </div>
                    `).join('');
                }

            } catch (err) {
                alert(err.message);
            }
        }

        function openPromoteModal() {
            document.getElementById('promote-modal').classList.remove('hidden');
            document.getElementById('promote-modal').classList.add('flex');
        }

        function closePromoteModal() {
            document.getElementById('promote-modal').classList.add('hidden');
            document.getElementById('promote-modal').classList.remove('flex');
        }

        async function executePromote() {
            const name = document.getElementById('input-promote-name').value.trim();
            const errDiv = document.getElementById('promote-error');
            errDiv.classList.add('hidden');

            try {
                const res = await fetch(`/api/trainings/${encodeURIComponent(trainingId)}/promote`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ target_name: name })
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || 'Erro ao promover modelo.');
                alert(data.message || 'Modelo promovido com sucesso!');
                closePromoteModal();
            } catch (err) {
                errDiv.innerText = err.message;
                errDiv.classList.remove('hidden');
            }
        }

        function escapeHtml(text) {
            if (!text) return '';
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        initTrainingPage();
    </script>
</body>
</html>"""

    @app.get("/api/workspace")
    def api_workspace_info():
        return {
            "root": str(workspace.root),
            "sources_root": str(workspace.sources_root),
            "versions_root": str(workspace.versions_root),
            "campaigns_root": str(workspace.sources_root),
            "releases_root": str(workspace.versions_root),
            "videos_root": str(workspace.videos_root),
        }

    @app.get("/api/models")
    def api_models_list():
        return list_available_models(workspace)

    @app.get("/api/trainings")
    def api_trainings_list():
        runs_dir = workspace.root / "runs" / "detect"
        if not runs_dir.exists():
            return []
        items = []
        registered_models = list_registered_models(workspace)
        for path in sorted(runs_dir.iterdir(), reverse=True):
            if path.is_dir():
                best = path / "weights" / "best.pt"
                args_yaml = path / "args.yaml"
                registry_path = run_registry_path(workspace, path.name)
                registry_run = (
                    load_yaml(registry_path) if registry_path.is_file() else {}
                )
                model_name = "YOLO"
                if args_yaml.exists():
                    try:
                        args = load_yaml(args_yaml)
                        model_name = args.get("model", "YOLO")
                    except Exception:
                        pass
                persisted_status = None
                workflow_path = path / "workflow_job.json"
                if workflow_path.is_file():
                    try:
                        persisted_status = json.loads(
                            workflow_path.read_text(encoding="utf-8")
                        ).get("status")
                    except (OSError, json.JSONDecodeError):
                        persisted_status = None
                items.append(
                    {
                        "name": path.name,
                        "status": registry_run.get("status")
                        or persisted_status
                        or ("completed" if best.is_file() else "unknown"),
                        "model": model_name,
                        "best": str(best) if best.is_file() else None,
                        "state": registry_run.get("state"),
                        "dataset_id": registry_run.get("dataset_id"),
                        "initial_model_id": registry_run.get("initial_model_id"),
                        "output_model_id": registry_run.get("output_model_id"),
                        "output_model": registered_models.get(
                            registry_run.get("output_model_id"), {}
                        ),
                    }
                )
        return items

    @app.delete("/api/trainings/{training_id}")
    def api_delete_training(training_id: str, confirm: str = ""):
        if confirm != training_id:
            raise HTTPException(status_code=400, detail="Confirmação divergente do ID do treinamento.")
        try:
            remove_training(training_id)
            return {"status": "ok", "message": f"Treinamento {training_id} excluído com sucesso."}
        except WorkflowError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/api/trainings/{training_id}")
    def api_training_detail(training_id: str):
        runs_dir = workspace.root / "runs" / "detect" / training_id
        if not runs_dir.exists():
            raise HTTPException(status_code=404, detail="Treinamento não encontrado.")

        best_path = runs_dir / "weights" / "best.pt"
        last_path = runs_dir / "weights" / "last.pt"
        log_path = runs_dir / "train.log"
        csv_path = runs_dir / "results.csv"
        args_yaml = runs_dir / "args.yaml"

        log_content = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""

        # Parser dos hiperparâmetros de entrada (args.yaml)
        training_args = {}
        if args_yaml.exists():
            try:
                training_args = load_yaml(args_yaml)
            except Exception:
                pass

        workflow_job = {}
        workflow_job_path = runs_dir / "workflow_job.json"
        if workflow_job_path.is_file():
            try:
                workflow_job = json.loads(workflow_job_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                workflow_job = {}
        registry_path = run_registry_path(workspace, training_id)
        registry_run = (
            load_yaml(registry_path) if registry_path.is_file() else {}
        )

        # Parser do relatório da release associada
        release_info = {}
        try:
            release_id = (workflow_job.get("metadata") or {}).get("release_id")
            if not release_id:
                release_id = training_args.get("name") if training_args.get("name") in list_versions(workspace) else None
            if not release_id and training_args.get("data"):
                try:
                    data_path = Path(str(training_args["data"])).resolve()
                    relative_data = data_path.relative_to(workspace.versions_root.resolve())
                    candidate = relative_data.parts[0]
                    if candidate in list_versions(workspace):
                        release_id = candidate
                except (OSError, ValueError, IndexError):
                    pass
            if not release_id:
                raise WorkflowError("Treinamento sem version_id persistido.")
            rel_status = version_status(workspace, release_id)
            release_info = {
                "release_id": rel_status["release_id"],
                "sources": rel_status.get("sources", []),
                "assignments": rel_status.get("assignments", {}),
                "evaluation_level": rel_status.get("evaluation_level", "legacy"),
                "build_report": rel_status.get("build_report") or {},
            }
        except Exception:
            pass
        if not release_info and registry_run.get("dataset_id"):
            dataset_path = dataset_registry_path(
                workspace, registry_run["dataset_id"]
            )
            if dataset_path.is_file():
                dataset = load_yaml(dataset_path)
                release_info = {
                    "release_id": dataset["dataset_id"],
                    "dataset_id": dataset["dataset_id"],
                    "sources": dataset.get("sources", []),
                    "assignments": dataset.get("splits", {}),
                    "evaluation_level": dataset.get(
                        "evaluation_level", "legacy"
                    ),
                    "build_report": {
                        "images": dataset.get("images"),
                        "boxes": dataset.get("boxes"),
                        "splits": dataset.get("splits", {}),
                        "manifest_sha256": dataset.get("manifest_sha256"),
                    },
                    "provenance": dataset.get("provenance", {}),
                }

        # Parser do CSV de resultados do YOLO (results.csv)
        metrics = {
            "completed_epochs": 0,
            "precision": None,
            "recall": None,
            "map50": None,
            "map50_95": None,
            "val_box_loss": None,
            "val_cls_loss": None,
            "val_dfl_loss": None,
        }
        if csv_path.exists():
            try:
                lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
                if len(lines) > 1:
                    headers = [h.strip() for h in lines[0].split(",")]
                    last_line = [v.strip() for v in lines[-1].split(",")]
                    row = dict(zip(headers, last_line))

                    metrics["completed_epochs"] = int(row.get("epoch", len(lines) - 1))
                    
                    # Tenta ler métricas padrão da Ultralytics
                    for k, v in row.items():
                        lk = k.lower()
                        if "precision" in lk:
                            metrics["precision"] = float(v)
                        elif "recall" in lk:
                            metrics["recall"] = float(v)
                        elif "map50-95" in lk or "map95" in lk:
                            metrics["map50_95"] = float(v)
                        elif "map50" in lk:
                            metrics["map50"] = float(v)
                        elif "val/box_loss" in lk:
                            metrics["val_box_loss"] = float(v)
                        elif "val/cls_loss" in lk:
                            metrics["val_cls_loss"] = float(v)
                        elif "val/dfl_loss" in lk:
                            metrics["val_dfl_loss"] = float(v)
            except Exception as exc:
                metrics["parse_error"] = str(exc)

        # Cálculo do tempo de execução baseado nos logs/arquivos
        duration_seconds = None
        if log_path.exists():
            try:
                stat_start = log_path.stat().st_ctime
                stat_end = (best_path if best_path.exists() else log_path).stat().st_mtime
                duration_seconds = max(0, int(stat_end - stat_start))
            except Exception:
                pass

        # Lista imagens de gráficos e validação disponíveis
        image_files = []
        for img in sorted(runs_dir.glob("*.png")):
            image_files.append(img.name)

        registered_model = list_registered_models(workspace).get(
            registry_run.get("output_model_id"), {}
        )
        registered_best = None
        for stored_path in registered_model.get("paths", []):
            candidate = workspace.resolve_path(stored_path)
            if candidate.is_file() and candidate.suffix.lower() == ".pt":
                registered_best = candidate
                break
        effective_best = best_path if best_path.is_file() else registered_best

        return {
            "id": training_id,
            "name": training_id,
            "status": registry_run.get(
                "status", "completed" if effective_best else "in_progress"
            ),
            "state": registry_run.get("state"),
            "has_best": effective_best is not None,
            "has_last": last_path.is_file(),
            "best_path": str(effective_best) if effective_best else None,
            "last_path": str(last_path) if last_path.is_file() else None,
            "args": training_args,
            "release": release_info,
            "metrics": metrics,
            "evaluations": registry_run.get("evaluations", {}),
            "robustness": registry_run.get("robustness", {}),
            "duration_seconds": duration_seconds,
            "log": log_content[-20000:],
            "images": image_files,
            "registry": registry_run,
            "registered_model": registered_model,
        }

    @app.post("/api/trainings/{training_id}/promote")
    def api_promote_model(training_id: str, req: PromoteModelReq = Body(default_factory=PromoteModelReq)):
        runs_dir = workspace.root / "runs" / "detect" / training_id
        best_path = runs_dir / "weights" / "best.pt"
        if not best_path.is_file():
            registry_path = run_registry_path(workspace, training_id)
            registry_run = (
                load_yaml(registry_path) if registry_path.is_file() else {}
            )
            model = list_registered_models(workspace).get(
                registry_run.get("output_model_id"), {}
            )
            best_path = next(
                (
                    workspace.resolve_path(path)
                    for path in model.get("paths", [])
                    if workspace.resolve_path(path).is_file()
                    and workspace.resolve_path(path).suffix.lower() == ".pt"
                ),
                best_path,
            )
        if not best_path.is_file():
            raise HTTPException(status_code=400, detail="Este treinamento não possui o modelo 'best.pt' finalizado para promover.")

        target_name = (req.target_name or f"{training_id}_best.pt").strip()
        if not target_name.endswith(".pt"):
            target_name += ".pt"
        if Path(target_name).name != target_name:
            raise HTTPException(
                status_code=400,
                detail="Informe somente o nome do arquivo, sem diretórios.",
            )

        dest_dir = workspace.models_root
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / target_name
        if dest_path.exists():
            if sha256(dest_path) != sha256(best_path) and not req.overwrite:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Já existe um modelo diferente com esse nome. "
                        "Escolha outro nome ou confirme a substituição explicitamente."
                    ),
                )
        if not dest_path.exists() or sha256(dest_path) != sha256(best_path):
            shutil.copy2(best_path, dest_path)
        promoted = promote_registered_model(workspace, training_id, dest_path)
        deployment = export_deployment_bundle(
            workspace,
            promoted["model_id"],
            artifact_path=dest_path,
        )
        deployment_manifest = (
            workspace.deployments_root
            / deployment["deployment_id"]
            / "deployment_manifest.yaml"
        )
        rel_dest = dest_path.relative_to(workspace.root).as_posix()
        return {
            "status": "ok",
            "message": (
                f"Modelo promovido para '{rel_dest}' e bundle imutável criado "
                f"em '{deployment_manifest.relative_to(workspace.root).as_posix()}'."
            ),
            "promoted_name": target_name,
            "path": rel_dest,
            "model_id": promoted["model_id"],
            "sha256": promoted["sha256"],
            "deployment_id": deployment["deployment_id"],
            "deployment_manifest": deployment_manifest.relative_to(
                workspace.root
            ).as_posix(),
        }

    @app.post("/api/models/{model_id}/deploy")
    def api_deploy_model(
        model_id: str,
        req: DeployModelReq = Body(default_factory=DeployModelReq),
    ):
        try:
            manifest = export_deployment_bundle(
                workspace,
                model_id,
                deployment_id=req.deployment_id,
                artifact_path=req.artifact_path,
            )
        except WorkflowError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        manifest_path = (
            workspace.deployments_root
            / manifest["deployment_id"]
            / "deployment_manifest.yaml"
        )
        return {
            "status": "ok",
            "deployment": manifest,
            "manifest_path": manifest_path.relative_to(workspace.root).as_posix(),
        }

    @app.get("/api/registry/status")
    def api_registry_status():
        return registry_status(workspace)

    @app.get("/api/registry/models")
    def api_registry_models():
        return {
            "models": list_registered_models(workspace),
            "aliases": list_registered_aliases(workspace),
        }

    @app.get("/api/registry/sources")
    def api_registry_sources():
        return list_registered_sources(workspace)

    @app.get("/api/sources")
    @app.get("/api/campaigns")
    def api_list_sources():
        return list_sources(workspace)

    @app.post("/api/sources")
    @app.post("/api/campaigns")
    def api_create_source(req: SourceCreateReq):
        try:
            path = create_source(
                workspace,
                source_id=req.target_id,
                videos_dir=Path(req.videos_dir),
                video_pattern=req.video_pattern,
                video_files=req.video_files or None,
                video_notes=req.video_notes or None,
                capture_units=req.capture_units or None,
                annotation={"classes": req.classes},
            )
            return {"status": "ok", "path": str(path)}
        except WorkflowError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/sources/upload")
    @app.post("/api/campaigns/upload")
    async def api_create_source_upload(
        source_id: str = Form(None),
        campaign_id: str = Form(None),
        classes: str = Form('["objeto"]'),
        video_notes: str = Form("{}"),
        capture_units: str = Form("[]"),
        videos: list[UploadFile] = File(...),
    ):
        try:
            target_id = source_id or campaign_id
            if not target_id:
                raise HTTPException(status_code=400, detail="Identificador da origem obrigatório.")
            validate_id(target_id, "source_id")
            class_list = json.loads(classes)
            notes_dict = json.loads(video_notes) if video_notes else {}
            units_list = json.loads(capture_units) if capture_units else []
            videos_dir = workspace.videos_root
            videos_dir.mkdir(parents=True, exist_ok=True)
            target_videos_dir = videos_dir / target_id
            if target_videos_dir.exists() or workspace.source_root(target_id).exists():
                raise WorkflowError(f"A origem ja existe: {target_id}")
            staging_dir = Path(tempfile.mkdtemp(prefix=f".{target_id}-", dir=videos_dir))
            video_filenames = []
            try:
                for file in videos:
                    filename = file.filename or ""
                    if not filename or Path(filename).name != filename:
                        raise WorkflowError(f"Nome de video invalido: {filename}")
                    if filename in video_filenames:
                        raise WorkflowError(f"Video duplicado no upload: {filename}")
                    dest = staging_dir / filename
                    with dest.open("wb") as handle:
                        shutil.copyfileobj(file.file, handle)
                    video_filenames.append(filename)
                staging_dir.replace(target_videos_dir)
                try:
                    path = create_source(
                        workspace,
                        source_id=target_id,
                        videos_dir=target_videos_dir,
                        video_pattern="*.mp4",
                        video_files=video_filenames,
                        video_notes=notes_dict,
                        capture_units=units_list or None,
                        annotation={"classes": class_list},
                    )
                except Exception:
                    shutil.rmtree(target_videos_dir, ignore_errors=True)
                    raise
            finally:
                if staging_dir.exists():
                    shutil.rmtree(staging_dir, ignore_errors=True)
            return {"status": "ok", "path": str(path)}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/sources/{source_id}")
    @app.get("/api/campaigns/{source_id}")
    def api_source_status(source_id: str):
        try:
            return source_status(workspace, source_id)
        except WorkflowError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/api/sources/{source_id}/finished-tasks")
    @app.get("/api/campaigns/{source_id}/finished-tasks")
    def api_inspect_finished_tasks(source_id: str):
        try:
            return inspect_finished_tasks(workspace, source_id)
        except WorkflowError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/sources/{source_id}/extract")
    @app.post("/api/campaigns/{source_id}/extract")
    def api_extract_frames(source_id: str, req: ExtractionReq | None = None):
        try:
            st = source_status(workspace, source_id)
            if st.get("frames", 0) > 0:
                raise WorkflowError("A segunda etapa (extração de frames) já foi concluída e não pode ser alterada.")
            if req is not None:
                source = load_source(workspace, source_id)
                extraction = req.model_dump()
                if req.mode == "smart":
                    if not req.model:
                        raise WorkflowError("O modo inteligente exige um modelo.")
                    model_path = workspace.resolve_path(req.model).resolve()
                    try:
                        model_path.relative_to(workspace.models_root.resolve())
                    except ValueError as exc:
                        raise WorkflowError("O modelo deve estar dentro de models/.") from exc
                    if not model_path.is_file():
                        raise WorkflowError(f"Modelo nao encontrado: {model_path}")
                else:
                    extraction["model"] = None
                source["extraction"] = extraction
                dump_yaml(workspace.source_config_path(source_id), source)
                register_source_manifest(workspace, source_id)
            manifest_path = extract_source_frames(workspace, source_id)
            return {"status": "ok", "manifest": str(manifest_path)}
        except WorkflowError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/sources/{source_id}/import-tasks")
    @app.post("/api/campaigns/{source_id}/import-tasks")
    def api_build_import_tasks(source_id: str, req: ImportTasksReq | None = None):
        try:
            st = source_status(workspace, source_id)
            if st.get("import_tasks", 0) > 0:
                raise WorkflowError("A terceira etapa (geração do import_tasks.json) já foi concluída e não pode ser alterada.")
            request = req or ImportTasksReq()
            if request.mode == "model":
                if not request.model:
                    raise WorkflowError("Selecione um modelo para pre-anotar.")
                model_path = workspace.resolve_path(request.model).resolve()
                try:
                    model_path.relative_to(workspace.models_root.resolve())
                except ValueError as exc:
                    raise WorkflowError("O modelo deve estar dentro de models/.") from exc
                if not model_path.is_file():
                    raise WorkflowError(f"Modelo nao encontrado: {model_path}")
                preannotate_source_frames(
                    workspace, source_id, model_path, confidence=request.confidence
                )
                source = load_source(workspace, source_id)
                source["annotation"]["backend"] = "local"
                source["annotation"]["model"] = request.model
                dump_yaml(workspace.source_config_path(source_id), source)
                register_source_manifest(workspace, source_id)
            output = build_import_tasks(
                workspace,
                source_id,
                include_predictions=request.mode != "none",
            )
            register_source_manifest(workspace, source_id)
            return {"status": "ok", "output": str(output)}
        except WorkflowError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/sources/{source_id}/start-label-studio")
    @app.post("/api/campaigns/{source_id}/start-label-studio")
    def api_start_label_studio(
        source_id: str,
        req: LabelStudioStartReq = Body(default_factory=LabelStudioStartReq),
    ):
        try:
            load_source(workspace, source_id)
            ml_job = None
            if req.enable_ml:
                ml_job = start_ml_backend_job(
                    job_manager, workspace, source_id, model_name=req.model, port=9090
                )
                if not wait_for_ml_backend(9090, timeout=20.0):
                    detail = "O backend de predicao nao ficou saudavel."
                    if isinstance(ml_job, dict) and ml_job.get("id"):
                        try:
                            failed_job = job_manager.get(ml_job["id"])
                            if failed_job.get("log"):
                                detail += f"\n{failed_job['log'][-2000:]}"
                        except WorkflowError:
                            pass
                    raise WorkflowError(detail)

            ls_job = start_label_studio_job(job_manager, workspace, source_id, port=8080)

            online = wait_for_port(8080, timeout=15.0)
            if not online:
                raise WorkflowError("O Label Studio nao respondeu na porta 8080.")
            integration_status = label_studio_integration_status(
                workspace, source_id
            )
            integration = None
            if integration_status["credentials"]["configured"]:
                integration = ensure_label_studio_project(
                    workspace,
                    source_id,
                    allow_partial_predictions=req.allow_partial_predictions,
                    ml_backend_url=(
                        "http://127.0.0.1:9090" if req.enable_ml else None
                    ),
                )
            target_url = (
                integration["url"]
                if integration
                else integration_status["credentials"]["base_url"]
            )
            return {
                "status": "ok",
                "online": online,
                "url": target_url,
                "ls_job": ls_job if isinstance(ls_job, dict) else None,
                "ml_job": ml_job if isinstance(ml_job, dict) else None,
                "integration": integration or integration_status,
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/label-studio/settings")
    def api_label_studio_settings():
        return public_credentials_status()

    @app.post("/api/label-studio/settings")
    def api_save_label_studio_settings(req: LabelStudioSettingsReq):
        try:
            client = LabelStudioClient(req.base_url, req.api_key)
            user = client.authenticate()
            save_label_studio_credentials(req.base_url, req.api_key)
            return {
                "status": "ok",
                "settings": public_credentials_status(),
                "user": {
                    "id": user.get("id") if isinstance(user, dict) else None,
                    "email": user.get("email") if isinstance(user, dict) else None,
                },
            }
        except WorkflowError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.delete("/api/label-studio/settings")
    def api_delete_label_studio_settings():
        delete_label_studio_credentials()
        return {"status": "ok", "settings": public_credentials_status()}

    @app.get("/api/sources/{source_id}/label-studio")
    @app.get("/api/campaigns/{source_id}/label-studio")
    def api_label_studio_integration_status(source_id: str):
        try:
            return label_studio_integration_status(workspace, source_id)
        except WorkflowError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/sources/{source_id}/label-studio/prepare")
    @app.post("/api/campaigns/{source_id}/label-studio/prepare")
    def api_prepare_label_studio(
        source_id: str,
        req: LabelStudioPrepareReq = Body(default_factory=LabelStudioPrepareReq),
    ):
        try:
            if not wait_for_port(8080, timeout=1.0):
                raise WorkflowError(
                    "Inicie o Label Studio antes de preparar o projeto."
                )
            return ensure_label_studio_project(
                workspace,
                source_id,
                allow_partial_predictions=req.allow_partial_predictions,
            )
        except WorkflowError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/sources/{source_id}/accept-export")
    @app.post("/api/campaigns/{source_id}/accept-export")
    def api_accept_export(source_id: str, req: ExportAcceptReq):
        try:
            target_path = Path(req.path)
            revision_id = req.revision_id
            if not revision_id:
                clean_stem = re.sub(r"[^A-Za-z0-9_-]", "_", target_path.stem)
                revision_id = f"rev_{clean_stem}"
            accepted, report = accept_native_export(
                workspace,
                source_id,
                target_path,
                revision_id=revision_id,
                allow_pending=req.allow_pending,
            )
            return {"status": "ok", "accepted": str(accepted), "report": str(report), "revision_id": revision_id}
        except WorkflowError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/versions")
    @app.get("/api/releases")
    def api_list_versions():
        return list_versions(workspace)

    @app.post("/api/versions/preview-split")
    @app.post("/api/releases/preview-split")
    def api_preview_split(req: SplitPreviewReq):
        try:
            src_id = req.source_id or req.campaign_id
            if not src_id:
                raise HTTPException(status_code=400, detail="Identificador da origem obrigatório.")
            return preview_split_metrics(
                workspace,
                src_id,
                req.assignments,
                req.revision_id,
                req.evaluation_level,
            )
        except WorkflowError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/versions")
    @app.post("/api/releases")
    def api_create_version(req: VersionCreateReq):
        try:
            path = create_version(
                workspace,
                version_id=req.target_id,
                source_ids=req.target_sources,
                assignments=req.assignments,
                annotation_revisions=req.annotation_revisions or None,
                evaluation_level=req.evaluation_level,
            )
            return {"status": "ok", "path": str(path)}
        except WorkflowError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/versions/{version_id}/build")
    @app.post("/api/releases/{version_id}/build")
    def api_build_version(version_id: str):
        try:
            manifest = build_version(workspace, version_id)
            return {"status": "ok", "manifest": str(manifest)}
        except WorkflowError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/versions/{version_id}")
    @app.get("/api/releases/{version_id}")
    def api_version_status(version_id: str):
        try:
            return version_status(workspace, version_id)
        except WorkflowError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/api/deletion-impact/{resource_type}/{resource_id}")
    def api_deletion_impact(resource_type: str, resource_id: str, source_id: str | None = None):
        try:
            return deletion_impact(resource_type, resource_id, source_id)
        except WorkflowError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.delete("/api/sources/{source_id}")
    @app.delete("/api/campaigns/{source_id}")
    def api_delete_source(
        source_id: str,
        confirm: str = "",
        cascade: bool = False,
        delete_videos: bool = True,
    ):
        try:
            if confirm != source_id:
                raise WorkflowError("Confirmação divergente do ID da origem.")
            impact = deletion_impact("source", source_id)
            if cascade:
                for version_id in impact["dependent_versions"]:
                    remove_version_with_dependents(version_id, cascade=True)
            delete_source(workspace, source_id, delete_video_files=delete_videos)
            return {"status": "ok", "message": f"Origem {source_id} excluída com sucesso.", "impact": impact, "cascade": cascade}
        except WorkflowError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.delete("/api/versions/{version_id}")
    @app.delete("/api/releases/{version_id}")
    def api_delete_version(version_id: str, confirm: str = "", cascade: bool = False):
        try:
            if confirm != version_id:
                raise WorkflowError("Confirmação divergente do ID da versão.")
            impact = deletion_impact("version", version_id)
            remove_version_with_dependents(version_id, cascade=cascade)
            return {"status": "ok", "message": f"Release {version_id} excluída com sucesso.", "impact": impact, "cascade": cascade}
        except WorkflowError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.delete("/api/sources/{source_id}/revisions/{revision_id}")
    def api_delete_revision(source_id: str, revision_id: str, confirm: str = "", cascade: bool = False):
        try:
            if confirm != revision_id:
                raise WorkflowError("Confirmação divergente do ID da revisão.")
            impact = deletion_impact("revision", revision_id, source_id)
            if cascade:
                for version_id in impact["dependent_versions"]:
                    remove_version_with_dependents(version_id, cascade=True)
            revision_root = workspace.source_root(source_id) / "label_studio" / "revisions" / revision_id
            validate_id(revision_id, "revision_id")
            if not revision_root.is_dir():
                raise WorkflowError(f"Revisão não encontrada: {revision_id}")
            shutil.rmtree(revision_root, ignore_errors=False)
            return {"status": "ok", "message": f"Revisão {revision_id} excluída com sucesso.", "impact": impact, "cascade": cascade}
        except WorkflowError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/versions/{version_id}/start-train")
    @app.post("/api/releases/{version_id}/start-train")
    def api_start_train(version_id: str, req: TrainStartReq):
        try:
            if importlib.util.find_spec("ultralytics") is None:
                raise WorkflowError(
                    "Ultralytics nao esta instalado no ambiente do Dataset Studio. "
                    "Execute: uv sync --all-extras"
                )
            train_timestamp_id = f"t_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"
            model_reference = resolve_model_reference(workspace, req.model)
            params = TrainingParams(
                model=model_reference,
                epochs=req.epochs,
                imgsz=req.imgsz,
                batch=req.batch,
                workers=req.workers,
                device=req.device,
                patience=req.patience,
                lr0=req.lr0,
                optimizer=req.optimizer,
                project=str(workspace.root / "runs" / "detect"),
                name=train_timestamp_id,
                extra_args={"exist_ok": True},
            )
            recipe = training_recipe(workspace, version_id, params)
            begin_training_record(
                workspace,
                train_timestamp_id,
                version_id,
                params,
            )
            job = job_manager.enqueue_training(
                command=recipe["command"],
                target=version_id,
                cwd=workspace.root,
                log_path=workspace.root / "runs" / "detect" / train_timestamp_id / "train.log",
                metadata={
                    "training_id": train_timestamp_id,
                    "release_id": version_id,
                    "model": model_reference,
                    "model_reference": req.model,
                    "epochs": req.epochs,
                    "imgsz": req.imgsz,
                },
                on_complete=lambda completed_job: finalize_training_record(
                    workspace,
                    train_timestamp_id,
                    completed_job["status"],
                ),
            )
            return job
        except WorkflowError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/jobs")
    def api_list_jobs():
        return job_manager.list()

    @app.get("/api/jobs/{job_id}")
    def api_get_job(job_id: str):
        try:
            return job_manager.get(job_id)
        except WorkflowError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.post("/api/jobs/{job_id}/stop")
    def api_stop_job(job_id: str):
        try:
            return job_manager.stop(job_id)
        except WorkflowError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/jobs/{job_id}/cancel")
    def api_cancel_job(job_id: str):
        try:
            return job_manager.cancel_queued(job_id)
        except WorkflowError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    runs_dir = workspace.root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/runs", StaticFiles(directory=runs_dir), name="runs")

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Servidor Web Local do Dataset Studio.")
    parser.add_argument("--host", default="127.0.0.1", help="Host do servidor web.")
    parser.add_argument("--port", type=int, default=8000, help="Porta do servidor web.")
    parser.add_argument("--workspace", type=Path, default=Path("."), help="Raiz do workspace.")
    parser.add_argument("--no-browser", action="store_true", help="Não abre automaticamente o navegador.")
    return parser.parse_args()


def main() -> None:
    import threading
    import webbrowser

    args = parse_args()
    ws = Workspace.from_path(args.workspace)
    app = create_web_app(ws)

    url = f"http://{args.host}:{args.port}/"
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    print("\n============================================================")
    print(" DATASET STUDIO - PAINEL INTERATIVO LOCAL")
    print(f" Servidor rodando em: {url}")
    print(f" Workspace: {ws.root}")
    print("============================================================\n")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
