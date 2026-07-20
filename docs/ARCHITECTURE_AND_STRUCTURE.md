# Estrutura e Arquitetura do Repositório - Dataset Studio 📐

O **Dataset Studio** adota uma **Arquitetura Limpa e Modular** (Ports & Adapters / Hexagonal) com separação rígida entre as regras de negócio do domínio, adaptadores externos, camada de aplicação e interfaces de usuário (CLI e Web).

---

## 1. Visão Geral da Arquitetura

```
+-----------------------------------------------------------------------+
|                         INTERFACES DE USUÁRIO                         |
|   CLI (cli/main.py)              |        Web Panel (web/app.py)     |
+------------------------------------+----------------------------------+
                                     |
                                     v
+-----------------------------------------------------------------------+
|                          CAMADA DE APLICAÇÃO                          |
|   JobManager (job_service.py)      |        CampaignService           |
|   ReleaseService                   |        TrainingRecipe            |
+------------------------------------+----------------------------------+
                                     |
                                     v
+------------------------------------+----------------------------------+
|                           DOMÍNIO CENTRAL                             |
|   Workspace                        |        Campaigns & Manifests     |
|   Annotations & Revisions          |        Releases & Splits         |
+------------------------------------+----------------------------------+
                                     |
                                     v
+-----------------------------------------------------------------------+
|                        PORTAS E ADAPTADORES                           |
|   Ports: Predictor, Trainer                                           |
|   Adapters: OpenCV (media), Ultralytics (YOLO), Label Studio Backend  |
+-----------------------------------------------------------------------+
```

---

## 2. Árvore de Diretórios e Onde Vai Cada Arquivo

Abaixo está a estrutura completa de pastas do repositório `dataset-studio`:

```
dataset-studio/
├── pyproject.toml               # Configurações do pacote Hatchling/uv e dependências
├── LICENSE                      # Licença aberta (MIT)
├── README.md                    # Visão geral do repositório e atalhos rápidos
├── dataset-studio.py            # Ponto de entrada executável para abrir o painel web
│
├── docs/                        # Documentação oficial do projeto
│   ├── USER_MANUAL.md           # Manual do Usuário com guia passo-a-passo
│   ├── ARCHITECTURE_AND_STRUCTURE.md # Este documento de arquitetura e estrutura
│   └── DOCUMENTATION_TODO.md    # Checklist e backlog de tarefas de documentação
│
├── models/                      # Diretório contendo modelos pré-treinados (.pt)
│
├── runs/                        # Saída de treinamentos do YOLO (padrão runs/detect/<release_id>/)
│   └── detect/
│       └── <release_id>/        # Logs, pesos (best.pt, last.pt) e métricas do treino
│
├── src/                         # Código-fonte Python do pacote dataset_studio
│   └── dataset_studio/
│       ├── __init__.py
│       │
│       ├── domain/              # Regras de Negócio Puramente Decoupled
│       │   ├── __init__.py
│       │   ├── workspace.py     # Abstração de Workspace (substitui caminhos fixos)
│       │   ├── campaigns.py     # Criação, leitura e estado de campanhas
│       │   ├── annotations.py   # Parsing, snapshots imutáveis e revisões de JSON
│       │   ├── releases.py      # Atribuição de vídeos a splits e materialização
│       │   └── errors.py        # Exceção central WorkflowError
│       │
│       ├── ports/               # Interfaces e Protocolos (Ports)
│       │   ├── __init__.py
│       │   ├── predictor.py     # Protocolo neutro Predictor
│       │   └── trainer.py       # Protocolo Trainer e dataclass TrainingParams
│       │
│       ├── adapters/            # Implementações Concretas (Adapters)
│       │   ├── opencv/
│       │   │   └── media.py     # Leitura de vídeos, amostragem e extração de frames
│       │   ├── ultralytics/
│       │   │   ├── predictor.py # Adaptador de inferência YOLO da Ultralytics
│       │   │   └── trainer.py   # Montador de comandos e receitas de treino YOLO
│       │   └── label_studio/
│       │       ├── ml_backend.py # Servidor ML Backend de detecção local (porta 9090)
│       │       ├── runner.py    # Runner de variáveis de ambiente do Label Studio
│       │       └── process_supervisor.py # Gerenciador de processos filhos/grupos
│       │
│       ├── application/         # Serviços de Orquestração da Aplicação
│       │   ├── __init__.py
│       │   ├── campaign_service.py # Status de campanhas e inspeção de finished_tasks
│       │   ├── release_service.py  # Status de releases e calculadora de splits
│       │   └── job_service.py      # JobManager para execução asíncrona de jobs/logs
│       │
│       ├── cli/                 # Interface de Linha de Comando (CLI)
│       │   └── main.py          # Entrypoint dataset-studio CLI
│       │
│       ├── web/                 # Interface Web Interativa (FastAPI + Tailwind)
│       │   └── app.py           # Servidor web, rotas REST e páginas HTML incorporadas
│       │
│       └── utils/               # Utilitários auxiliares
│           └── repair_region_ids.py # Reparo de IDs no SQLite do Label Studio
│
└── tests/                       # Suíte de Testes Automatizados
    ├── sanity/
    ├── unit/                    # Testes unitários (ex: test_training.py)
    ├── characterization/        # Testes de caracterização do domínio
    └── integration/             # Testes de integração E2E, UI e JobManager
```

---

## 3. Diretórios de Dados Dinâmicos do Workspace

Durante o uso, o **Dataset Studio** lê e grava dados dentro da raiz do workspace configurado:

1. **`campaigns/<campaign_id>/`**:
   - `campaign.yaml`: Manifesto da campanha (vídeos, padrão, classes).
   - `frames/raw/images/`: Imagens extraídas dos vídeos.
   - `frames/frame_manifest.json`: Registro estruturado dos frames.
   - `label_studio/import_tasks.json`: Tarefas geradas para importação.
   - `label_studio/finished_tasks/`: Pasta onde o usuário salva o JSON final exportado pelo Label Studio.
   - `revisions/`: Snapshots imutáveis de revisões aceitas (`rev_auto`, `r001`).

2. **`dataset/releases/<release_id>/`**:
   - `release.yaml`: Manifesto de atribuição de vídeos por split.
   - `data.yaml`: Arquivo de configuração YOLO gerado na materialização.
   - `manifest.csv`: Manifesto tabular da release.
   - `train/` e `val/`: Pastas físicas com subdiretórios `images/` e `labels/`.

3. **`models/`**:
   - Modelos `.pt` pré-treinados colocados pelo usuário para uso na extração inteligente, pré-anotação ou ML Backend.

4. **`runs/detect/<release_id>/`**:
   - Diretório de saída do treinamento YOLO (logs, gráficos e modelo treinado `best.pt`).
