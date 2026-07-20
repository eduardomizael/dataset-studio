"""Aplicação Web Local FastAPI do Dataset Studio."""

from __future__ import annotations

import argparse
import atexit
import json
import shutil
import uvicorn
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from dataset_studio.adapters.label_studio.runner import (
    start_label_studio_job,
    start_ml_backend_job,
    wait_for_port,
)
from dataset_studio.adapters.opencv.media import extract_campaign_frames
from dataset_studio.application import (
    JobManager,
    TrainingParams,
    campaign_status,
    inspect_finished_tasks,
    list_available_models,
    preview_split_metrics,
    release_status,
    training_recipe,
)
from dataset_studio.domain import (
    WorkflowError,
    Workspace,
    accept_native_export,
    build_import_tasks,
    build_release,
    create_campaign,
    create_release,
    list_campaigns,
    list_releases,
    load_campaign,
)

job_manager = JobManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Ao fechar o servidor do Dataset Studio, encerra os outros servidores (Label Studio e ML Backend)
    job_manager.stop_all(wait=True)


atexit.register(lambda: job_manager.stop_all(wait=False))


class LabelStudioStartReq(BaseModel):
    enable_ml: bool = False
    model: str | None = None


class CampaignCreateReq(BaseModel):
    campaign_id: str
    videos_dir: str = "videos"
    video_pattern: str = "*.mp4"
    video_files: list[str] | None = None
    classes: list[str] = Field(default_factory=lambda: ["objeto"])


class ExportAcceptReq(BaseModel):
    path: str
    revision_id: str | None = None
    allow_pending: bool = False


class ReleaseCreateReq(BaseModel):
    release_id: str
    campaigns: list[str] = Field(min_length=1)
    assignments: dict[str, list[str]]
    annotation_revisions: dict[str, str] = Field(default_factory=dict)


class SplitPreviewReq(BaseModel):
    campaign_id: str
    assignments: dict[str, list[str]]
    revision_id: str | None = None


class TrainStartReq(BaseModel):
    model: str = "yolo26n.pt"
    epochs: int = 50
    imgsz: int = 640
    batch: int = -1
    workers: int = 0
    device: str = "auto"
    patience: int = 50
    lr0: float = 0.01
    optimizer: str = "auto"


def create_web_app(workspace: Workspace) -> FastAPI:
    app = FastAPI(title="Dataset Studio Web Dashboard", version="0.1.0", lifespan=lifespan)

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
                <p class="text-slate-400 text-sm mt-1">Gerenciamento autônomo do ciclo de vida de datasets para visão computacional</p>
            </div>
            <div class="flex items-center gap-3">
                <button onclick="openCreateCampaignModal()" class="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-sm rounded-lg shadow-lg shadow-indigo-600/30 transition">
                    + Nova Campanha
                </button>
                <span class="px-3 py-1.5 bg-slate-800 text-slate-400 border border-slate-700 rounded-lg text-xs font-mono">
                    Workspace: <span id="ws-root">...</span>
                </span>
            </div>
        </header>

        <!-- Layout 3 Colunas -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <!-- Coluna 1: Campanhas -->
            <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col gap-4">
                <div class="flex justify-between items-center pb-2 border-b border-slate-800">
                    <h2 class="text-xl font-bold text-slate-200 flex items-center gap-2">
                        <span>📁</span> Campanhas
                    </h2>
                    <button onclick="loadData()" class="text-xs text-indigo-400 hover:underline">🔄 Atualizar</button>
                </div>
                <div id="campaigns-list" class="space-y-4">
                    <p class="text-slate-500 text-sm">Carregando campanhas...</p>
                </div>
            </div>

            <!-- Coluna 2: Releases -->
            <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col gap-4">
                <div class="flex justify-between items-center pb-2 border-b border-slate-800">
                    <h2 class="text-xl font-bold text-slate-200 flex items-center gap-2">
                        <span>📦</span> Releases Materializadas
                    </h2>
                    <button onclick="loadData()" class="text-xs text-indigo-400 hover:underline">🔄 Atualizar</button>
                </div>
                <div id="releases-list" class="space-y-4">
                    <p class="text-slate-500 text-sm">Carregando releases...</p>
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

    <!-- Modal Nova Campanha -->
    <div id="modal-campaign" class="fixed inset-0 bg-slate-950/80 backdrop-blur-sm hidden flex items-center justify-center p-4 z-50">
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 max-w-lg w-full shadow-2xl space-y-5">
            <div class="flex justify-between items-center pb-3 border-b border-slate-800">
                <h3 class="text-lg font-bold text-indigo-400">Criar Nova Campanha</h3>
                <button onclick="closeCreateCampaignModal()" class="text-slate-400 hover:text-white">✕</button>
            </div>
            
            <div class="space-y-4 text-sm">
                <div>
                    <label class="block text-slate-300 font-medium mb-1">ID da Campanha</label>
                    <input type="text" id="input-campaign-id" placeholder="ex: campanha_01" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2.5 text-white focus:border-indigo-500 focus:outline-none">
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

                <div>
                    <label class="block text-slate-300 font-medium mb-1">Classes de Objetos (separadas por vírgula)</label>
                    <input type="text" id="input-classes" value="peixe" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2.5 text-white focus:border-indigo-500 focus:outline-none">
                </div>
            </div>

            <div id="modal-error" class="text-rose-400 text-xs hidden"></div>

            <div class="flex justify-end gap-3 pt-3 border-t border-slate-800">
                <button onclick="closeCreateCampaignModal()" class="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg font-medium text-xs">Cancelar</button>
                <button onclick="submitCreateCampaign()" class="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg font-medium text-xs shadow-lg shadow-indigo-600/30">Criar Campanha</button>
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
            if (files && files.length > 0) {
                const names = Array.from(files).map(f => f.name).join(', ');
                infoDiv.innerText = `✓ ${files.length} vídeo(s) selecionado(s): ${names}`;
                infoDiv.classList.remove('hidden');
            } else {
                infoDiv.classList.add('hidden');
            }
        }

        async function submitCreateCampaign() {
            const id = document.getElementById('input-campaign-id').value.trim();
            const filesInput = document.getElementById('input-video-files');
            const classesRaw = document.getElementById('input-classes').value.trim();
            const errDiv = document.getElementById('modal-error');

            if (!id) {
                errDiv.innerText = 'Preencha o ID da campanha.';
                errDiv.classList.remove('hidden');
                return;
            }
            if (!filesInput.files || filesInput.files.length === 0) {
                errDiv.innerText = 'Selecione pelo menos um arquivo de vídeo.';
                errDiv.classList.remove('hidden');
                return;
            }

            const classes = classesRaw ? classesRaw.split(',').map(c => c.trim()).filter(Boolean) : ['objeto'];

            const formData = new FormData();
            formData.append('campaign_id', id);
            formData.append('classes', JSON.stringify(classes));
            for (const file of filesInput.files) {
                formData.append('videos', file);
            }

            try {
                errDiv.innerText = 'Enviando vídeos e criando campanha...';
                errDiv.classList.remove('hidden');
                errDiv.classList.remove('text-rose-400');
                errDiv.classList.add('text-indigo-400');

                const res = await fetch('/api/campaigns/upload', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                if (!res.ok) {
                    throw new Error(data.detail || 'Erro ao criar campanha.');
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

                // Campanhas
                const resC = await fetch('/api/campaigns');
                const campaigns = await resC.json();
                const divC = document.getElementById('campaigns-list');
                
                if (campaigns.length === 0) {
                    divC.innerHTML = `
                        <div class="text-center py-8 border border-dashed border-slate-800 rounded-xl">
                            <p class="text-slate-400 text-sm font-medium">Nenhuma campanha criada ainda.</p>
                            <button onclick="openCreateCampaignModal()" class="mt-3 text-xs text-indigo-400 font-semibold hover:underline">
                                + Clique aqui para criar a primeira campanha
                            </button>
                        </div>
                    `;
                } else {
                    let html = '';
                    for (const cId of campaigns) {
                        const resSt = await fetch(`/api/campaigns/${cId}`);
                        const st = await resSt.json();
                        
                        html += `
                            <a href="/campaign.html?id=${st.campaign_id}" class="block p-5 bg-slate-800/40 border border-slate-800 rounded-xl space-y-2 hover:border-indigo-500/50 hover:bg-slate-800/70 transition group">
                                <div class="flex justify-between items-start">
                                    <div>
                                        <h3 class="font-bold text-indigo-300 text-lg group-hover:text-indigo-200">${st.campaign_id}</h3>
                                        <div class="text-xs text-slate-400 mt-0.5">
                                            Vídeos: <span class="text-slate-200 font-mono">${st.videos}</span> | 
                                            Frames: <span class="text-slate-200 font-mono">${st.frames}</span> | 
                                            Tasks: <span class="text-slate-200 font-mono">${st.import_tasks}</span>
                                        </div>
                                    </div>
                                    <span class="px-2.5 py-1 bg-slate-800 text-indigo-400 border border-indigo-500/30 rounded-md text-xs font-semibold">
                                        Etapa: ${st.next_action}
                                    </span>
                                </div>
                                <div class="text-xs text-indigo-400 font-medium pt-1 flex items-center gap-1 group-hover:translate-x-1 transition-transform">
                                    <span>Ver detalhes e etapas</span> &rarr;
                                </div>
                            </a>
                        `;
                    }
                    divC.innerHTML = html;
                }

                // Releases
                const resR = await fetch('/api/releases');
                const releases = await resR.json();
                const divR = document.getElementById('releases-list');

                if (releases.length === 0) {
                    divR.innerHTML = '<p class="text-slate-500 text-sm py-4 text-center">Nenhuma release materializada ainda.</p>';
                } else {
                    divR.innerHTML = releases.map(r => `
                        <a href="/release.html?id=${r}" class="block p-4 bg-slate-800/40 border border-slate-800 rounded-xl flex justify-between items-center hover:border-emerald-500/50 hover:bg-slate-800/70 transition group">
                            <div>
                                <div class="font-bold text-emerald-400 group-hover:text-emerald-300">${r}</div>
                                <div class="text-xs text-slate-400">Clique para ver detalhes, splits e treinar</div>
                            </div>
                            <span class="px-2.5 py-1 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded-md text-xs font-semibold">Materializada &rarr;</span>
                        </a>
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
                        <div class="p-4 bg-slate-800/40 border border-slate-800 rounded-xl flex justify-between items-center">
                            <div>
                                <div class="font-bold text-amber-400">${t.name}</div>
                                <div class="text-xs text-slate-400">Modelo: ${t.model || 'N/A'} | Status: ${t.status}</div>
                            </div>
                            <span class="px-2.5 py-1 bg-amber-500/10 text-amber-400 border border-amber-500/20 rounded-md text-xs font-semibold">${t.status}</span>
                        </div>
                    `).join('');
                }

            } catch (err) {
                console.error(err);
            }
        }
        loadData();
    </script>
</body>
</html>"""

    @app.get("/campaign.html", response_class=HTMLResponse)
    def campaign_detail_page():
        return """<!DOCTYPE html>
<html lang="pt-BR" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Studio - Detalhes da Campanha</title>
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
                <h1 class="text-3xl font-extrabold tracking-tight text-indigo-400" id="camp-title">Campanha</h1>
                <p class="text-slate-400 text-sm mt-1" id="camp-subtitle">Carregando informações da campanha...</p>
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

                    <!-- Painel de Métricas da Exportação Detectada -->
                    <div id="finished-metrics-panel" class="hidden bg-slate-900 border border-emerald-500/40 rounded-xl p-4 space-y-3">
                        <div class="flex justify-between items-center">
                            <h4 class="font-bold text-emerald-400 text-sm flex items-center gap-2">
                                <span>✓</span> Exportação Detectada e Concluída!
                            </h4>
                            <span id="metrics-filename" class="text-xs font-mono text-slate-400 bg-slate-800 px-2 py-1 rounded">...</span>
                        </div>

                        <div class="grid grid-cols-2 md:grid-cols-4 gap-3 text-center">
                            <div class="p-3 bg-slate-800/60 rounded-lg border border-slate-700/60">
                                <div class="text-slate-400 text-xs">Total Imagens</div>
                                <div class="text-xl font-bold text-indigo-300 font-mono" id="m-tasks">0</div>
                            </div>
                            <div class="p-3 bg-slate-800/60 rounded-lg border border-slate-700/60">
                                <div class="text-slate-400 text-xs">Total Anotações</div>
                                <div class="text-xl font-bold text-emerald-300 font-mono" id="m-boxes">0</div>
                            </div>
                            <div class="p-3 bg-slate-800/60 rounded-lg border border-slate-700/60">
                                <div class="text-slate-400 text-xs">Negativos Confirmados</div>
                                <div class="text-xl font-bold text-amber-300 font-mono" id="m-negs">0</div>
                            </div>
                            <div class="p-3 bg-slate-800/60 rounded-lg border border-slate-700/60">
                                <div class="text-slate-400 text-xs">Classes</div>
                                <div class="text-xs font-bold text-purple-300 font-mono mt-1" id="m-classes">-</div>
                            </div>
                        </div>

                        <div class="pt-2 flex justify-end">
                            <button onclick="createReleaseFromCampaign()" class="px-5 py-2.5 bg-purple-600 hover:bg-purple-500 text-white font-medium text-xs rounded-lg shadow-lg shadow-purple-600/30 transition">
                                📦 Seguir para Criar Release &rarr;
                            </button>
                        </div>
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
                const res = await fetch(`/api/campaigns/${campaignId}/extract`, { method: 'POST' });
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
                const res = await fetch(`/api/campaigns/${campaignId}/import-tasks`, { method: 'POST' });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || 'Erro ao gerar import tasks');
                alert('Arquivo import_tasks.json gerado com sucesso!');
                loadCampaignDetails();
            } catch (err) {
                alert(err.message);
            }
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

                if (data.url) {
                    if (labelStudioTab) {
                        labelStudioTab.location.href = data.url;
                    } else {
                        window.open(data.url, '_blank');
                    }
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
                }

                // finished_tasks info
                const fin = st.finished_info || {};
                document.getElementById('finished-tasks-path').innerText = fin.finished_tasks_dir || `campaigns/${campaignId}/label_studio/finished_tasks`;

                if (fin.found) {
                    document.getElementById('step-4-status').className = "text-xs font-semibold px-2.5 py-1 bg-emerald-500/10 text-emerald-400 rounded-md";
                    document.getElementById('step-4-status').innerText = "✓ Concluído";
                    document.getElementById('finished-metrics-panel').classList.remove('hidden');
                    document.getElementById('metrics-filename').innerText = fin.latest_file.name;

                    const m = fin.metrics || {};
                    document.getElementById('m-tasks').innerText = m.total_tasks || 0;
                    document.getElementById('m-boxes').innerText = m.total_boxes || 0;
                    document.getElementById('m-negs').innerText = m.confirmed_negatives || 0;

                    const clsEntries = Object.entries(m.class_counts || {});
                    document.getElementById('m-classes').innerText = clsEntries.length > 0
                        ? clsEntries.map(([k, v]) => `${k}: ${v}`).join(', ')
                        : 'Nenhuma caixa';
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

            } catch (err) {
                console.error(err);
            }
        }

        loadCampaignDetails();
    </script>
</body>
</html>"""

    @app.get("/release.html", response_class=HTMLResponse)
    def release_detail_page():
        return """<!DOCTYPE html>
<html lang="pt-BR" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Studio - Detalhes da Release & Treinamento</title>
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
                <h1 class="text-3xl font-extrabold tracking-tight text-emerald-400" id="rel-title">Release</h1>
                <p class="text-slate-400 text-sm mt-1" id="rel-subtitle">Divisão de dataset por vídeos completos (sem vazamento) e treinamento</p>
            </div>
            <span id="rel-status-badge" class="px-3 py-1.5 bg-slate-800 text-emerald-400 border border-emerald-500/30 rounded-lg text-xs font-semibold">
                Status: ...
            </span>
        </header>

        <!-- Seção 1: Divisão dos Vídeos e Calculadora de Splits -->
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
            <h2 class="text-xl font-bold text-slate-100 flex items-center gap-2 border-b border-slate-800 pb-3">
                <span>🎥</span> Atribuição dos Vídeos ao Dataset (Train / Val)
            </h2>

            <!-- Cards de Métricas em Tempo Real -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div class="p-5 bg-slate-800/60 border-2 border-indigo-500/50 rounded-xl space-y-2">
                    <div class="flex justify-between items-center">
                        <span class="font-bold text-indigo-300 text-base">📊 Conjunto de Treino (Train)</span>
                        <span class="text-xs bg-indigo-500/20 text-indigo-300 font-mono px-2 py-0.5 rounded" id="cnt-train-videos">0 Vídeos</span>
                    </div>
                    <div class="flex justify-around pt-2 border-t border-slate-700/50 text-center">
                        <div>
                            <div class="text-xs text-slate-400">Frames</div>
                            <div class="text-xl font-bold text-indigo-200 font-mono" id="cnt-train-frames">0</div>
                        </div>
                        <div>
                            <div class="text-xs text-slate-400">Anotações</div>
                            <div class="text-xl font-bold text-emerald-300 font-mono" id="cnt-train-boxes">0</div>
                        </div>
                    </div>
                </div>

                <div class="p-5 bg-slate-800/60 border-2 border-purple-500/50 rounded-xl space-y-2">
                    <div class="flex justify-between items-center">
                        <span class="font-bold text-purple-300 text-base">📊 Conjunto de Validação (Val)</span>
                        <span class="text-xs bg-purple-500/20 text-purple-300 font-mono px-2 py-0.5 rounded" id="cnt-val-videos">0 Vídeos</span>
                    </div>
                    <div class="flex justify-around pt-2 border-t border-slate-700/50 text-center">
                        <div>
                            <div class="text-xs text-slate-400">Frames</div>
                            <div class="text-xl font-bold text-purple-200 font-mono" id="cnt-val-frames">0</div>
                        </div>
                        <div>
                            <div class="text-xs text-slate-400">Anotações</div>
                            <div class="text-xl font-bold text-amber-300 font-mono" id="cnt-val-boxes">0</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Tabela de Atribuição por Vídeo -->
            <div class="space-y-3">
                <h3 class="text-sm font-bold text-slate-300">Defina o papel de cada vídeo:</h3>
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
        <div class="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
            <h2 class="text-xl font-bold text-amber-400 flex items-center gap-2 border-b border-slate-800 pb-3">
                <span>⚡</span> Treinamento do Modelo YOLO
            </h2>

            <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div>
                    <label class="block text-slate-300 font-medium mb-1">Modelo de Partida</label>
                    <select id="train-model-select" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2.5 text-white text-xs">
                        <option value="yolo26n.pt">yolo26n.pt (Novo Modelo Base)</option>
                        <option value="yolov8n.pt">yolov8n.pt (Modelo YOLOv8 Nano)</option>
                    </select>
                </div>

                <div>
                    <label class="block text-slate-300 font-medium mb-1">Épocas (epochs)</label>
                    <input type="number" id="train-epochs-input" value="50" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2.5 text-white">
                </div>

                <div>
                    <label class="block text-slate-300 font-medium mb-1">Tamanho Imagem (imgsz)</label>
                    <input type="number" id="train-imgsz-input" value="640" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2.5 text-white">
                </div>
            </div>

            <div>
                <button onclick="startTrainingProcess()" id="btn-start-train" class="px-6 py-3 bg-amber-600 hover:bg-amber-500 text-white font-bold text-xs rounded-xl shadow-lg shadow-amber-600/30 transition">
                    🚀 Iniciar Treinamento
                </button>
            </div>

            <!-- Terminal em Tempo Real -->
            <div id="terminal-section" class="hidden space-y-3">
                <div class="flex justify-between items-center">
                    <h3 class="font-bold text-slate-200 text-sm flex items-center gap-2">
                        <span class="w-2.5 h-2.5 rounded-full bg-amber-400 animate-pulse"></span>
                        Monitoramento do Treinamento em Tempo Real
                    </h3>
                    <span id="job-status-tag" class="text-xs font-mono text-amber-300 bg-amber-500/20 px-2 py-0.5 rounded">Executando...</span>
                </div>
                <pre id="terminal-logs" class="p-4 bg-slate-950 rounded-xl border border-slate-800 text-xs font-mono text-slate-300 h-64 overflow-y-auto">Iniciando job...</pre>
            </div>
        </div>

    </div>

    <script>
        const urlParams = new URLSearchParams(window.location.search);
        const releaseId = urlParams.get('id');
        const campaignId = urlParams.get('campaign');

        let videoList = [];
        let videoAssignments = {};

        async function updateSplitPreview() {
            try {
                const res = await fetch('/api/releases/preview-split', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        campaign_id: campaignId,
                        assignments: {
                            train: videoList.filter(v => videoAssignments[v] === 'train').map(v => campaignId + '/' + v),
                            val: videoList.filter(v => videoAssignments[v] === 'val').map(v => campaignId + '/' + v)
                        }
                    })
                });
                const data = await res.json();
                
                document.getElementById('cnt-train-videos').innerText = `${data.train.videos} Vídeo(s)`;
                document.getElementById('cnt-train-frames').innerText = data.train.frames;
                document.getElementById('cnt-train-boxes').innerText = data.train.boxes;

                document.getElementById('cnt-val-videos').innerText = `${data.val.videos} Vídeo(s)`;
                document.getElementById('cnt-val-frames').innerText = data.val.frames;
                document.getElementById('cnt-val-boxes').innerText = data.val.boxes;
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
                const trainV = videoList.filter(v => videoAssignments[v] === 'train').map(v => campaignId + '/' + v);
                const valV = videoList.filter(v => videoAssignments[v] === 'val').map(v => campaignId + '/' + v);

                // Criar release
                await fetch('/api/releases', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        release_id: releaseId,
                        campaigns: [campaignId],
                        assignments: { train: trainV, val: valV }
                    })
                });

                // Materializar
                const resB = await fetch(`/api/releases/${releaseId}/build`, { method: 'POST' });
                const dataB = await resB.json();
                if (!resB.ok) throw new Error(dataB.detail || 'Erro na materialização');

                alert('Dataset materializado com sucesso!');
                document.getElementById('rel-status-badge').innerText = 'Status: Materializada';
            } catch (err) {
                alert(err.message);
            }
        }

        async function startTrainingProcess() {
            const model = document.getElementById('train-model-select').value;
            const epochs = parseInt(document.getElementById('train-epochs-input').value);
            const imgsz = parseInt(document.getElementById('train-imgsz-input').value);

            try {
                const res = await fetch(`/api/releases/${releaseId}/start-train`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: model,
                        epochs: epochs,
                        imgsz: imgsz
                    })
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || 'Erro ao iniciar treinamento');

                document.getElementById('terminal-section').classList.remove('hidden');
                pollJob(data.job_id);
            } catch (err) {
                alert(err.message);
            }
        }

        async function pollJob(jobId) {
            const logBox = document.getElementById('terminal-logs');
            const statusTag = document.getElementById('job-status-tag');

            const interval = setInterval(async () => {
                try {
                    const res = await fetch(`/api/jobs/${jobId}`);
                    const job = await res.json();
                    
                    statusTag.innerText = `Status: ${job.status}`;
                    logBox.innerText = job.log || 'Aguardando logs...';
                    logBox.scrollTop = logBox.scrollHeight;

                    if (job.status === 'completed' || job.status === 'failed') {
                        clearInterval(interval);
                        alert(`Treinamento finalizado com status: ${job.status}`);
                    }
                } catch (err) {
                    console.error(err);
                }
            }, 2000);
        }

        async function initReleasePage() {
            document.getElementById('rel-title').innerText = releaseId || 'Release';
            
            try {
                const resSt = await fetch(`/api/campaigns/${campaignId}`);
                const st = await resSt.json();
                const report = st.annotation_report || {};
                videoList = Object.keys(report.per_video || {});

                if (videoList.length === 0) {
                    document.getElementById('video-assignment-list').innerHTML = '<p class="text-slate-500 text-xs">Nenhum vídeo encontrado.</p>';
                    return;
                }

                // Atribuir por padrão 80% train, 20% val
                const mid = Math.ceil(videoList.length * 0.8);
                videoList.forEach((v, idx) => {
                    videoAssignments[v] = (idx < mid) ? 'train' : 'val';
                });
                if (videoList.length === 1) videoAssignments[videoList[0]] = 'train';

                // Renderizar tabela
                let html = '';
                videoList.forEach(v => {
                    const role = videoAssignments[v];
                    html += `
                        <div class="p-3 bg-slate-800/40 border border-slate-800 rounded-lg flex justify-between items-center text-xs">
                            <span class="font-mono text-slate-200">${v}</span>
                            <div class="flex gap-4">
                                <label class="flex items-center gap-1 cursor-pointer">
                                    <input type="radio" name="role_${v}" value="train" ${role==='train'?'checked':''} onchange="setVideoRole('${v}', 'train')" class="text-indigo-600">
                                    <span class="text-indigo-300 font-medium">Train</span>
                                </label>
                                <label class="flex items-center gap-1 cursor-pointer">
                                    <input type="radio" name="role_${v}" value="val" ${role==='val'?'checked':''} onchange="setVideoRole('${v}', 'val')" class="text-purple-600">
                                    <span class="text-purple-300 font-medium">Val</span>
                                </label>
                            </div>
                        </div>
                    `;
                });
                document.getElementById('video-assignment-list').innerHTML = html;
                updateSplitPreview();

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

            } catch (err) {
                console.error(err);
            }
        }

        initReleasePage();
    </script>
</body>
</html>"""

    @app.get("/api/workspace")
    def api_workspace_info():
        return {
            "root": str(workspace.root),
            "campaigns_root": str(workspace.campaigns_root),
            "releases_root": str(workspace.releases_root),
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
        for path in sorted(runs_dir.iterdir(), reverse=True):
            if path.is_dir():
                best = path / "weights" / "best.pt"
                items.append(
                    {
                        "name": path.name,
                        "status": "completed" if best.is_file() else "in_progress",
                        "model": "YOLO",
                        "best": str(best) if best.is_file() else None,
                    }
                )
        return items

    @app.get("/api/campaigns")
    def api_list_campaigns():
        return list_campaigns(workspace)

    @app.post("/api/campaigns")
    def api_create_campaign(req: CampaignCreateReq):
        try:
            path = create_campaign(
                workspace,
                campaign_id=req.campaign_id,
                videos_dir=Path(req.videos_dir),
                video_pattern=req.video_pattern,
                video_files=req.video_files or None,
                annotation={"classes": req.classes},
            )
            return {"status": "ok", "path": str(path)}
        except WorkflowError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/campaigns/upload")
    async def api_create_campaign_upload(
        campaign_id: str = Form(...),
        classes: str = Form('["objeto"]'),
        videos: list[UploadFile] = File(...),
    ):
        try:
            class_list = json.loads(classes)
            videos_dir = workspace.videos_root
            videos_dir.mkdir(parents=True, exist_ok=True)
            video_filenames = []
            for file in videos:
                dest = videos_dir / file.filename
                with dest.open("wb") as handle:
                    shutil.copyfileobj(file.file, handle)
                video_filenames.append(file.filename)
            path = create_campaign(
                workspace,
                campaign_id=campaign_id,
                videos_dir=videos_dir,
                video_pattern="*.mp4",
                video_files=video_filenames,
                annotation={"classes": class_list},
            )
            return {"status": "ok", "path": str(path)}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/campaigns/{campaign_id}")
    def api_campaign_status(campaign_id: str):
        try:
            return campaign_status(workspace, campaign_id)
        except WorkflowError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/api/campaigns/{campaign_id}/finished-tasks")
    def api_inspect_finished_tasks(campaign_id: str):
        try:
            return inspect_finished_tasks(workspace, campaign_id)
        except WorkflowError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/campaigns/{campaign_id}/extract")
    def api_extract_frames(campaign_id: str):
        try:
            st = campaign_status(workspace, campaign_id)
            if st.get("frames", 0) > 0:
                raise WorkflowError("A segunda etapa (extração de frames) já foi concluída e não pode ser alterada.")
            manifest_path = extract_campaign_frames(workspace, campaign_id)
            return {"status": "ok", "manifest": str(manifest_path)}
        except WorkflowError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/campaigns/{campaign_id}/import-tasks")
    def api_build_import_tasks(campaign_id: str):
        try:
            st = campaign_status(workspace, campaign_id)
            if st.get("import_tasks", 0) > 0:
                raise WorkflowError("A terceira etapa (geração do import_tasks.json) já foi concluída e não pode ser alterada.")
            output = build_import_tasks(workspace, campaign_id)
            return {"status": "ok", "output": str(output)}
        except WorkflowError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/campaigns/{campaign_id}/start-label-studio")
    def api_start_label_studio(
        campaign_id: str,
        req: LabelStudioStartReq = Body(default_factory=LabelStudioStartReq),
    ):
        try:
            load_campaign(workspace, campaign_id)
            ls_job = start_label_studio_job(job_manager, workspace, campaign_id, port=8080)
            ml_job = None
            if req.enable_ml:
                ml_job = start_ml_backend_job(
                    job_manager, workspace, campaign_id, model_name=req.model, port=9090
                )
                wait_for_port(9090, timeout=8.0)

            online = wait_for_port(8080, timeout=10.0)
            return {
                "status": "ok",
                "online": online,
                "url": "http://127.0.0.1:8080",
                "ls_job": ls_job if isinstance(ls_job, dict) else None,
                "ml_job": ml_job if isinstance(ml_job, dict) else None,
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/campaigns/{campaign_id}/accept-export")
    def api_accept_export(campaign_id: str, req: ExportAcceptReq):
        try:
            accepted, report = accept_native_export(
                workspace,
                campaign_id,
                Path(req.path),
                revision_id=req.revision_id,
                allow_pending=req.allow_pending,
            )
            return {"status": "ok", "accepted": str(accepted), "report": str(report)}
        except WorkflowError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/releases")
    def api_list_releases():
        return list_releases(workspace)

    @app.post("/api/releases/preview-split")
    def api_preview_split(req: SplitPreviewReq):
        try:
            return preview_split_metrics(workspace, req.campaign_id, req.assignments, req.revision_id)
        except WorkflowError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/releases")
    def api_create_release(req: ReleaseCreateReq):
        try:
            path = create_release(
                workspace,
                release_id=req.release_id,
                campaign_ids=req.campaigns,
                assignments=req.assignments,
                annotation_revisions=req.annotation_revisions or None,
            )
            return {"status": "ok", "path": str(path)}
        except WorkflowError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/releases/{release_id}/build")
    def api_build_release(release_id: str):
        try:
            manifest = build_release(workspace, release_id)
            return {"status": "ok", "manifest": str(manifest)}
        except WorkflowError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/releases/{release_id}")
    def api_release_status(release_id: str):
        try:
            return release_status(workspace, release_id)
        except WorkflowError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.post("/api/releases/{release_id}/start-train")
    def api_start_train(release_id: str, req: TrainStartReq):
        try:
            params = TrainingParams(
                model=req.model,
                epochs=req.epochs,
                imgsz=req.imgsz,
                batch=req.batch,
                workers=req.workers,
                device=req.device,
                patience=req.patience,
                lr0=req.lr0,
                optimizer=req.optimizer,
                project=str(workspace.root / "runs" / "detect"),
                name=release_id,
            )
            recipe = training_recipe(workspace, release_id, params)
            job = job_manager.start(
                command=recipe["command"],
                kind="training",
                target=release_id,
                cwd=workspace.root,
                log_path=workspace.root / "runs" / "detect" / release_id / "train.log",
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

    print(f"\n============================================================")
    print(f" DATASET STUDIO - PAINEL INTERATIVO LOCAL")
    print(f" Servidor rodando em: {url}")
    print(f" Workspace: {ws.root}")
    print(f"============================================================\n")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
