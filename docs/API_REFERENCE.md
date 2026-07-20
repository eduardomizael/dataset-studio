# Referência da API REST - Dataset Studio 🌐

A interface web do **Dataset Studio** é suportada por uma API REST desenvolvida em **FastAPI**. Esta API local gerencia as operações do workspace, o ciclo de vida das campanhas (origens), a materialização de releases (versões) e os treinamentos YOLO.

Por padrão, a API roda localmente em `http://127.0.0.1:8000/api`.

---

## 1. Workspace

### `GET /api/workspace`
Retorna informações estruturais e caminhos absolutos do workspace de dados atualmente ativo.

* **Resposta de Sucesso (`200 OK`)**:
  ```json
  {
    "root": "D:\\dataset-studio",
    "sources_root": "D:\\dataset-studio\\campaigns",
    "versions_root": "D:\\dataset-studio\\dataset\\releases",
    "campaigns_root": "D:\\dataset-studio\\campaigns",
    "releases_root": "D:\\dataset-studio\\dataset\\releases",
    "videos_root": "D:\\dataset-studio\\videos"
  }
  ```

---

## 2. Modelos e Treinamentos

### `GET /api/models`
Lista todos os modelos pré-treinados (`.pt`) disponíveis na pasta `models/` que podem ser usados para extração inteligente ou ML Backend.

* **Resposta de Sucesso (`200 OK`)**:
  ```json
  [
    "yolov8n.pt",
    "yolo26n.pt"
  ]
  ```

### `GET /api/trainings`
Lista os treinamentos YOLO (concluídos ou em execução) encontrados na pasta `runs/detect/`.

* **Resposta de Sucesso (`200 OK`)**:
  ```json
  [
    {
      "name": "release_01",
      "status": "completed",
      "model": "YOLO",
      "best": "D:\\dataset-studio\\runs\\detect\\release_01\\weights\\best.pt"
    }
  ]
  ```

---

## 3. Origens de Dados / Campanhas (`sources` ou `campaigns`)

> [!NOTE]
> Por questões de retrocompatibilidade e facilidade de transição de termos técnicos, todas as rotas de origens de dados aceitam os prefixos `/api/sources` e `/api/campaigns`.

### `GET /api/sources` | `GET /api/campaigns`
Retorna uma lista com os identificadores das origens de dados criadas no workspace.

* **Resposta de Sucesso (`200 OK`)**:
  ```json
  ["origem_peixes_01", "origem_peixes_02"]
  ```

### `POST /api/sources` | `POST /api/campaigns`
Cria uma nova estrutura de origem de dados (sem upload físico imediato de arquivos).

* **Corpo da Requisição (JSON)**:
  ```json
  {
    "source_id": "origem_peixes_01",
    "videos_dir": "videos",
    "video_pattern": "*.mp4",
    "classes": ["peixe", "detrito"]
  }
  ```
* **Resposta de Sucesso (`200 OK`)**:
  ```json
  {
    "status": "ok",
    "path": "D:\\dataset-studio\\campaigns\\origem_peixes_01"
  }
  ```

### `POST /api/sources/upload` | `POST /api/campaigns/upload`
Cria uma origem de dados recebendo fisicamente os arquivos de vídeo via formulário multipart. Salva os vídeos no diretório `videos/` do workspace.

* **Corpo da Requisição (`multipart/form-data`)**:
  * `source_id` (string, opcional): ID da origem.
  * `campaign_id` (string, opcional): ID alternativo da origem.
  * `classes` (string, JSON): Lista de classes, ex: `'["peixe"]'`.
  * `videos` (arquivos): Um ou mais arquivos de vídeo.
* **Resposta de Sucesso (`200 OK`)**:
  ```json
  {
    "status": "ok",
    "path": "D:\\dataset-studio\\campaigns\\origem_peixes_01"
  }
  ```

### `GET /api/sources/{source_id}` | `GET /api/campaigns/{source_id}`
Retorna o estado detalhado de uma campanha/origem de dados, incluindo contagem de vídeos, frames e tarefas para o Label Studio.

* **Resposta de Sucesso (`200 OK`)**:
  ```json
  {
    "source_id": "origem_peixes_01",
    "videos": 3,
    "frames": 120,
    "import_tasks": 120,
    "next_action": "label_studio",
    "video_details": [
      {
        "name": "canaleta_fluxo_01.mp4",
        "size_human": "12.4 MB",
        "resolution": "1920x1080",
        "fps": 30.0
      }
    ],
    "finished_info": {
      "found": true,
      "latest_file": {
        "name": "project-1-at-2026-07-20-17-00.json"
      },
      "metrics": {
        "total_tasks": 120,
        "total_boxes": 345,
        "confirmed_negatives": 15,
        "class_counts": {
          "peixe": 345
        }
      }
    }
  }
  ```

### `POST /api/sources/{source_id}/extract` | `POST /api/campaigns/{source_id}/extract`
Dispara o processo síncrono de extração de frames dos vídeos da origem de dados para o disco.

* **Resposta de Sucesso (`200 OK`)**:
  ```json
  {
    "status": "ok",
    "manifest": "D:\\dataset-studio\\campaigns\\origem_peixes_01\\frames\\frame_manifest.json"
  }
  ```

### `POST /api/sources/{source_id}/import-tasks` | `POST /api/campaigns/{source_id}/import-tasks`
Gera o arquivo `import_tasks.json` estruturado, pronto para ser importado no Label Studio.

* **Resposta de Sucesso (`200 OK`)**:
  ```json
  {
    "status": "ok",
    "output": "D:\\dataset-studio\\campaigns\\origem_peixes_01\\label_studio\\import_tasks.json"
  }
  ```

### `POST /api/sources/{source_id}/start-label-studio` | `POST /api/campaigns/{source_id}/start-label-studio`
Inicializa o processo do Label Studio em segundo plano e, se solicitado, inicializa também o ML Backend (servidor de predições).

* **Corpo da Requisição (JSON)**:
  ```json
  {
    "enable_ml": true,
    "model": "yolo26n.pt"
  }
  ```
* **Resposta de Sucesso (`200 OK`)**:
  ```json
  {
    "status": "ok",
    "online": true,
    "url": "http://127.0.0.1:8080",
    "ls_job": { "job_id": "ls_job_123", "status": "running" },
    "ml_job": { "job_id": "ml_job_456", "status": "running" }
  }
  ```

### `POST /api/sources/{source_id}/accept-export` | `POST /api/campaigns/{source_id}/accept-export`
Consome e valida o arquivo JSON exportado do Label Studio, criando um snapshot imutável de revisão.

* **Corpo da Requisição (JSON)**:
  ```json
  {
    "path": "D:\\dataset-studio\\campaigns\\origem_peixes_01\\label_studio\\finished_tasks\\export.json",
    "revision_id": "r001",
    "allow_pending": false
  }
  ```
* **Resposta de Sucesso (`200 OK`)**:
  ```json
  {
    "status": "ok",
    "accepted": "D:\\dataset-studio\\campaigns\\origem_peixes_01\\revisions\\r001\\annotations.json",
    "report": "D:\\dataset-studio\\campaigns\\origem_peixes_01\\revisions\\r001\\revision_report.json"
  }
  ```

---

## 4. Versões de Dataset / Releases (`versions` ou `releases`)

### `GET /api/versions` | `GET /api/releases`
Retorna a lista de todas as versões materializadas do dataset.

* **Resposta de Sucesso (`200 OK`)**:
  ```json
  ["release_peixes_v1"]
  ```

### `POST /api/versions/preview-split` | `POST /api/releases/preview-split`
Calcula em tempo real a proporção de frames e de caixas de anotação caso o usuário aplique determinada atribuição de vídeos a splits (`train` / `val`).

* **Corpo da Requisição (JSON)**:
  ```json
  {
    "campaign_id": "origem_peixes_01",
    "assignments": {
      "train": ["origem_peixes_01/video1.mp4", "origem_peixes_01/video2.mp4"],
      "val": ["origem_peixes_01/video3.mp4"]
    },
    "revision_id": "r001"
  }
  ```
* **Resposta de Sucesso (`200 OK`)**:
  ```json
  {
    "train": {
      "videos": 2,
      "frames": 80,
      "boxes": 240
    },
    "val": {
      "videos": 1,
      "frames": 40,
      "boxes": 105
    }
  }
  ```

### `POST /api/versions` | `POST /api/releases`
Cria a configuração física e manifestos de uma nova versão do dataset com a divisão selecionada.

* **Corpo da Requisição (JSON)**:
  ```json
  {
    "release_id": "release_peixes_v1",
    "campaigns": ["origem_peixes_01"],
    "assignments": {
      "train": ["origem_peixes_01/video1.mp4", "origem_peixes_01/video2.mp4"],
      "val": ["origem_peixes_01/video3.mp4"]
    },
    "annotation_revisions": {
      "origem_peixes_01": "r001"
    }
  }
  ```
* **Resposta de Sucesso (`200 OK`)**:
  ```json
  {
    "status": "ok",
    "path": "D:\\dataset-studio\\dataset\\releases\\release_peixes_v1"
  }
  ```

### `POST /api/versions/{version_id}/build` | `POST /api/releases/{version_id}/build`
Materializa fisicamente o dataset estruturando as imagens e rótulos YOLO no disco (`train/images`, `train/labels`, etc.) e gerando o `data.yaml`.

* **Resposta de Sucesso (`200 OK`)**:
  ```json
  {
    "status": "ok",
    "manifest": "D:\\dataset-studio\\dataset\\releases\\release_peixes_v1\\manifest.csv"
  }
  ```

### `GET /api/versions/{version_id}` | `GET /api/releases/{version_id}`
Retorna informações de status e quantidade de dados na release materializada.

* **Resposta de Sucesso (`200 OK`)**:
  ```json
  {
    "release_id": "release_peixes_v1",
    "materialized": true,
    "classes": ["peixe"],
    "splits": {
      "train": { "images": 80, "labels": 80 },
      "val": { "images": 40, "labels": 40 }
    }
  }
  ```

### `POST /api/versions/{version_id}/start-train` | `POST /api/releases/{version_id}/start-train`
Inicia assincronamente em segundo plano o processo de treinamento YOLO a partir daquela release.

* **Corpo da Requisição (JSON)**:
  ```json
  {
    "model": "yolo26n.pt",
    "epochs": 50,
    "imgsz": 640,
    "batch": -1,
    "workers": 0,
    "device": "auto",
    "patience": 50,
    "lr0": 0.01,
    "optimizer": "auto"
  }
  ```
* **Resposta de Sucesso (`200 OK`)**:
  ```json
  {
    "job_id": "train_release_peixes_v1",
    "status": "running",
    "command": ["python", "-m", "ultralytics", "detect", "train", "..."],
    "log_path": "D:\\dataset-studio\\runs\\detect\\release_peixes_v1\\train.log"
  }
  ```

---

## 5. Jobs em Segundo Plano

### `GET /api/jobs`
Retorna a lista de todos os processos/jobs ativos executando no painel (YOLO, Label Studio, ML Backend).

* **Resposta de Sucesso (`200 OK`)**:
  ```json
  [
    {
      "job_id": "train_release_peixes_v1",
      "status": "running",
      "kind": "training"
    }
  ]
  ```

### `GET /api/jobs/{job_id}`
Retorna os metadados do job e a saída do terminal (logs) em tempo real.

* **Resposta de Sucesso (`200 OK`)**:
  ```json
  {
    "job_id": "train_release_peixes_v1",
    "status": "running",
    "kind": "training",
    "target": "release_peixes_v1",
    "log": "Epoch    gpu_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n 1/50         0G      1.245     0.8415      1.102         12        640: ..."
  }
  ```

### `POST /api/jobs/{job_id}/stop`
Envia um sinal de encerramento para o processo do job de forma segura.

* **Resposta de Sucesso (`200 OK`)**:
  ```json
  {
    "job_id": "train_release_peixes_v1",
    "status": "stopped"
  }
  ```
