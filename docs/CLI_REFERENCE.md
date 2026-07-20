# Guia de Referência do CLI - Dataset Studio 💻

O **Dataset Studio** possui uma interface de linha de comando (CLI) completa e intuitiva, permitindo automatizar tarefas de MLOps, criar campanhas, materializar datasets e disparar treinamentos de modelos YOLO a partir do terminal.

Para rodar os comandos a partir da raiz do repositório utilizando o gerenciador `uv`, utilize o prefixo `uv run dataset-studio` (ou execute a chamada direta ao módulo Python se instalado: `python -m dataset_studio.cli.main`).

---

## Opções Globais
* `--workspace <caminho>`: Caminho da raiz do seu workspace de dados. Se omitido, assume a pasta atual (`.`).

Exemplo:
```bash
uv run dataset-studio --workspace D:\meu_workspace source list
```

---

## 1. Subcomandos de Origens de Dados (`source` ou `campaign`)

Gerencie as campanhas/origens de dados no seu workspace.

### `source create`
Cria uma nova estrutura de pasta e manifesto YAML para uma campanha de dados.

* **Argumentos e Flags**:
  * `--id <id>` (Obrigatório): Identificador único da origem (ex: `origem_peixes_01`).
  * `--videos-dir <caminho>` (Opcional): Pasta onde se encontram os vídeos da campanha (padrão: `videos`).
  * `--pattern <glob>` (Opcional): Padrão glob para buscar vídeos (padrão: `*.mp4`).
  * `--classes <lista>` (Opcional): Lista de classes que serão rotuladas (padrão: `objeto`).

* **Exemplo**:
  ```bash
  uv run dataset-studio source create --id camp_peixes --videos-dir D:\videos_gravados --pattern "*.mp4" --classes peixe detrito_fisico
  ```

### `source list`
Lista todos os identificadores de origens de dados configuradas no workspace.

* **Exemplo**:
  ```bash
  uv run dataset-studio source list
  ```

### `source status`
Exibe os detalhes de status estruturado de uma origem de dados em formato JSON (quantidade de vídeos, frames extraídos, tarefas geradas, etc.).

* **Argumentos**:
  * `--id <id>` (Obrigatório): Identificador da origem.

* **Exemplo**:
  ```bash
  uv run dataset-studio source status --id camp_peixes
  ```

---

## 2. Geração de Tarefas de Importação (`build-import`)

### `build-import`
Lê os frames extraídos de uma origem e monta o arquivo `import_tasks.json` na pasta do Label Studio. Esse arquivo conterá referências de imagens locais e, se configurado, pré-anotações automáticas.

* **Argumentos**:
  * `--source <id>` (Obrigatório): Identificador da origem.

* **Exemplo**:
  ```bash
  uv run dataset-studio build-import --source camp_peixes
  ```

---

## 3. Revisões e Aceite de Anotações (`accept-revision`)

### `accept-revision`
Valida e consome um JSON final exportado nativamente do Label Studio (no formato JSON convencional do Label Studio) e cria uma revisão snapshot imutável dentro da campanha.

* **Argumentos**:
  * `--source <id>` (Obrigatório): Identificador da origem.
  * `--export <caminho_arquivo>` (Obrigatório): Caminho físico para o arquivo `.json` exportado do Label Studio.
  * `--revision-id <id>` (Opcional): Identificador da revisão (padrão: gerado de forma sequencial, ex: `r001`).
  * `--allow-pending` (Opcional): Flag que aceita a importação mesmo se houver tarefas ainda não completadas ou sem anotações no Label Studio.

* **Exemplo**:
  ```bash
  uv run dataset-studio accept-revision --source camp_peixes --export D:\exportacoes\project-1.json --revision-id rev_final --allow-pending
  ```

---

## 4. Subcomandos de Versões e Splits (`version` ou `release`)

Gerencie a criação de conjuntos de dados físicos baseados em splits por vídeo sem vazamento temporal.

### `version create`
Configura os arquivos lógicos de uma nova versão do dataset e atribui cada vídeo a um split específico (`train` ou `val`).

* **Argumentos**:
  * `--id <id>` (Obrigatório): ID da nova versão do dataset (ex: `dataset_v1`).
  * `--sources <lista_de_ids>` (Obrigatório): Um ou mais IDs de campanhas de origem a incluir.
  * `--assignments-json <json_string>` (Obrigatório): JSON que mapeia os vídeos a seus respectivos papéis (`train` ou `val`).

* **Exemplo**:
  ```bash
  uv run dataset-studio version create --id dataset_v1 --sources camp_peixes --assignments-json "{\"train\":[\"camp_peixes/canaleta_video1.mp4\"], \"val\":[\"camp_peixes/canaleta_video2.mp4\"]}"
  ```

### `version build`
Materializa fisicamente o dataset na pasta `dataset/releases/<version_id>/`. Copia fisicamente as imagens selecionadas, gera os labels no formato txt (YOLO) e constrói o arquivo descritor `data.yaml` para o treino.

* **Argumentos**:
  * `--id <id>` (Obrigatório): ID da versão/release a materializar.

* **Exemplo**:
  ```bash
  uv run dataset-studio version build --id dataset_v1
  ```

### `version list`
Lista todos os conjuntos de dados (versões) materializados ou configurados no workspace.

* **Exemplo**:
  ```bash
  uv run dataset-studio version list
  ```

### `version status`
Exibe os dados detalhados da versão materializada (contagem de imagens de treino/validação, classes registradas, etc.).

* **Argumentos**:
  * `--id <id>` (Obrigatório): ID da versão.

* **Exemplo**:
  ```bash
  uv run dataset-studio version status --id dataset_v1
  ```

### `version train`
Configura e executa (ou apenas exibe) a receita e comando CLI de treinamento do YOLO para a versão materializada do dataset.

* **Argumentos**:
  * `--id <id>` (Obrigatório): ID da versão materializada a treinar.
  * `--model <modelo>` (Opcional): Modelo de partida (padrão: `yolo26n.pt`).
  * `--epochs <int>` (Opcional): Quantidade de épocas (padrão: `50`).
  * `--imgsz <int>` (Opcional): Resolução das imagens de treinamento (padrão: `640`).
  * `--batch <int>` (Opcional): Batch size (padrão: `-1` para auto-batch size).
  * `--workers <int>` (Opcional): CPU workers para carregamento de dados (padrão: `0`).
  * `--device <device>` (Opcional): Dispositivo de hardware (padrão: `auto`).
  * `--patience <int>` (Opcional): Limite de épocas sem melhora para early stopping (padrão: `50`).
  * `--lr0 <float>` (Opcional): Taxa de aprendizado inicial (padrão: `0.01`).
  * `--optimizer <opt>` (Opcional): Otimizador, como SGD, AdamW ou auto (padrão: `auto`).
  * `--dry-run` (Opcional): Exibe o comando CLI completo do YOLO gerado com todos os parâmetros sem executá-lo fisicamente.

* **Exemplo (Apenas visualizar comando)**:
  ```bash
  uv run dataset-studio version train --id dataset_v1 --model yolov8n.pt --epochs 100 --dry-run
  ```

* **Exemplo (Executar Treinamento no console)**:
  ```bash
  uv run dataset-studio version train --id dataset_v1 --model yolov8n.pt --epochs 100
  ```
