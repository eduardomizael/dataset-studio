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
|   JobManager (job_service.py)      |        SourceService             |
|   VersionService                   |        TrainingRecipe            |
+------------------------------------+----------------------------------+
                                     |
                                     v
+------------------------------------+----------------------------------+
|                           DOMÍNIO CENTRAL                             |
|   Workspace                        |        Sources & Manifests       |
|   Annotations & Revisions          |        Versions & Splits         |
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
├── videos/                      # Vídeos isolados por origem
│   └── <source_id>/
├── dataset/
│   ├── sources/                 # Origens e revisões
│   ├── versions/                # Versões configuradas/materializadas
│   └── archive/                 # Snapshots legados deduplicados por SHA-256
├── deployments/                 # Bundles imutáveis para aplicações consumidoras
├── runs/                        # Saída de treinamentos do YOLO
│   └── detect/
│       └── <training_id>/       # workflow_job.json, logs, pesos e métricas
│
├── src/                         # Código-fonte Python do pacote dataset_studio
│   └── dataset_studio/
│       ├── __init__.py
│       │
│       ├── domain/              # Regras de Negócio Puramente Decoupled
│       │   ├── __init__.py
│       │   ├── workspace.py     # Abstração de Workspace (substitui caminhos fixos)
│       │   ├── sources.py       # Criação, fixação e remoção de origens
│       │   ├── annotations.py   # Parsing, snapshots imutáveis e revisões de JSON
│       │   ├── versions.py      # Splits, materialização transacional e remoção
│       │   ├── campaigns.py     # Alias legado para sources.py
│       │   ├── releases.py      # Alias legado para versions.py
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
│       │       ├── api_client.py  # Cliente da API oficial e autenticação automática
│       │       ├── credentials.py # Credencial única fora do repositório
│       │       ├── ml_backend.py # Servidor ML Backend de detecção local (porta 9090)
│       │       ├── runner.py    # Runner de variáveis de ambiente do Label Studio
│       │       └── process_supervisor.py # Gerenciador de processos filhos/grupos
│       │
│       ├── application/         # Serviços de Orquestração da Aplicação
│       │   ├── __init__.py
│       │   ├── source_service.py   # Status de origens e inspeção de finished_tasks
│       │   ├── label_studio_service.py # Vínculo, preflight e importação idempotente
│       │   ├── version_service.py  # Status de versões e calculadora de splits
│       │   ├── campaign_service.py # Alias legado
│       │   ├── release_service.py  # Alias legado
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

1. **`dataset/sources/<source_id>/`**:
   - `source.yaml`: Vídeos, hashes, extração, classes, backend de anotação e
     perfil de predição congelado.
   - `frames/raw/images/`: Imagens extraídas dos vídeos.
   - `frame_manifest.json`: Registro estruturado dos frames e predições.
   - `label_studio/import_tasks.json`: Tarefas geradas para importação.
   - `label_studio/finished_tasks/`: Pasta onde o usuário salva o JSON final exportado pelo Label Studio.
   - `label_studio/revisions/<revision_id>/`: Exportação aceita e relatório imutável.

2. **`dataset/versions/<version_id>/`**:
   - `version.yaml`: Revisões escolhidas e atribuição de vídeos aos quatro splits.
   - `data.yaml`: Arquivo de configuração YOLO gerado na materialização.
   - `data_test_stress.yaml`: Avaliação específica do split de estresse, quando existente.
   - `manifest.csv`: Manifesto tabular da release.
   - `build_report.json`: Contagens e hashes da materialização.
   - `images/<split>/` e `labels/<split>/`: Dados físicos em formato YOLO.

3. **`models/`**:
   - Modelos `.pt` pré-treinados colocados pelo usuário para uso na extração inteligente, pré-anotação ou ML Backend.

4. **`dataset/archive/`**:
   - `objects/<prefixo>/<sha256>`: conteúdo físico único e deduplicado.
   - `snapshots/<snapshot_id>/manifest.csv`: caminhos originais e hashes.
   - `snapshots/<snapshot_id>/snapshot.yaml`: identidade, origem e estatísticas.
   - O arquivo preserva material legado sem apresentá-lo como versão nativa.

5. **`deployments/<deployment_id>/`**:
   - `deployment_manifest.yaml`: modelo, dataset, run, hashes e estado.
   - `model.pt` ou `model_bundle/`: artefato autocontido para inferência.

6. **`runs/detect/<training_id>/`**:
   - `workflow_job.json`: Estado persistido, parâmetros e `version_id` de origem.
   - `train.log`, `results.csv`, `args.yaml`, gráficos e `weights/best.pt`.
   - `run.yaml`: Manifest consolidado do treinamento.
   - `evaluations/summary.json`: Métricas finais de `test_normal`,
     `test_stress` e queda de robustez.
   - `provenance/`: Snapshot da versão consumida pelo treinamento.

7. **`registry/`**:
   - É um catálogo derivado; os manifestos junto dos recursos continuam
     canônicos.
   - `sources/<source_id>.yaml`: Referência e hash do `source.yaml`.
   - `models.yaml`: Identidades lógicas, hashes, pais, origem e estado dos modelos.
   - `aliases.yaml`: Caminhos físicos associados a cada `model_id`.
   - `datasets/<dataset_id>.yaml`: Manifests nativos ou reconstruídos.
   - `runs/<training_id>.yaml`: Relação dataset → modelo inicial → checkpoint.
   - Registros reconstruídos preservam `origin` e `confidence`; nunca são
     apresentados como evidência produzida durante o treinamento.

---

## 4. Invariantes e transações

- Uploads são escritos em staging e publicados em `videos/<source_id>/` somente após validação.
- A existência de `import_tasks.json` fixa a origem e bloqueia reconstrução pelo domínio, API e interface.
- `label_studio/integration.json` registra o projeto vinculado, hashes e cobertura, mas não contém o token.
- A importação no Label Studio é idempotente: um projeto não vazio só é reutilizado quando sua origem e contagem de tarefas são compatíveis.
- A predição automática prioriza cobertura total; cobertura parcial exige confirmação explícita.
- Consultas `GET` não aceitam exportações nem criam revisões.
- Cada revisão é um snapshot explícito de uma exportação nativa do Label Studio.
- Uma versão exige todos os vídeos atribuídos exatamente uma vez entre `train`, `val`, `test_normal` e `test_stress`.
- A materialização ocorre em um diretório temporário. O diretório final só é substituído depois que todos os artefatos são concluídos.
- `manifest.csv` registra hashes da imagem de origem, imagem materializada e label; `build_report.json` registra os hashes do manifesto e da configuração.
- Antes do treinamento, a versão e o modelo inicial são fixados no registry por
  ID e SHA-256. Ao terminar, métricas e hashes dos checkpoints são consolidados.
- Depois do treino, o mesmo `best.pt` é avaliado em `test_normal` e
  `test_stress`. Nenhum teste participa da otimização ou da escolha do peso.
- Um alias não cria um novo modelo: pesos byte a byte idênticos compartilham o
  mesmo `model_id`.
- Exclusão é uma operação destrutiva explícita, separada da imutabilidade durante o ciclo normal.

## 5. Processos e autonomia

- O Label Studio usa a porta 8080 e recebe o workspace como document root para arquivos locais.
- A API oficial do Label Studio cria ou reconhece o projeto, configura a fila e evita ajustes repetitivos por origem.
- O ML Backend usa a porta 9090, valida `/health` antes de a API declarar
  sucesso e carrega modelo, classes, confiança, device e ROI do perfil
  congelado na origem. Arquivos em `config/` são templates usados apenas na
  criação; mudanças posteriores não alteram origens existentes.
- O treinador chama `sys.executable` do próprio Dataset Studio. Nenhum caminho para outro repositório é permitido.
- No Windows, o extra `cuda` resolve PyTorch pelo índice oficial CUDA 12.8 configurado em `pyproject.toml`.
