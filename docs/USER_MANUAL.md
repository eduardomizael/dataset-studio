# Manual do Usuário - Dataset Studio 🚀

O **Dataset Studio** é uma ferramenta autônoma projetada para organizar, revisar, materializar datasets de visão computacional e executar o treinamento de modelos YOLO com total controle e isolamento.

---

## 1. Inicializando a Aplicação

Para iniciar o **Dataset Studio** e abrir a interface web no navegador, execute o comando na raiz do repositório `dataset-studio`:

```bash
uv run dataset-studio.py
```

A aplicação abrirá automaticamente no endereço: `http://127.0.0.1:8000/`.

---

## 2. Tela Inicial (Dashboard)

A tela inicial exibe um painel dividido em 3 colunas principais:

1. **📁 Campanhas**: Lista de campanhas de anotação de dados criadas no workspace.
2. **📦 Releases Materializadas**: Releases montadas e prontas para treinamento.
3. **⚡ Treinamentos**: Lista de treinamentos executados ou em andamento na pasta `runs/detect/`.

No canto superior direito, há o botão **`+ Nova Campanha`** e a indicação da pasta raiz do **Workspace**.

---

## 3. Fluxo de uma Campanha (4 Etapas em Acordeão)

Ao clicar em qualquer campanha na Tela Inicial, você será direcionado para a tela de detalhes (`/campaign.html?id=NOME_DA_CAMPANHA`), onde o ciclo de vida é organizado em **4 etapas progressivas**:

### Etapa 1: Seleção dos Vídeos
- Ao criar a campanha no modal inicial, você seleciona um ou múltiplos arquivos de vídeo (`.mp4`, `.avi`, `.mkv`) através da caixa de seleção nativa do sistema operacional.
- O sistema copia/associa os vídeos e exibe a contagem na Etapa 1 com o status **`✓ Concluído`**.

---

### Etapa 2: Seleção do Modo de Extração dos Frames
Permite escolher como os frames dos vídeos serão extraídos para imagem:

- **Modo Uniforme (Sem Filtro)**:
  - Extrai 1 frame a cada N quadros definidos pelo usuário (`frame_step`, ex: 30 = 1 frame/segundo a 30fps).
  - Recomendado para amostragem limpa e regular de novos vídeos.
- **Modo Inteligente (Com Modelo)**:
  - Utiliza um modelo pré-treinado localizado na pasta `models/` para detectar regiões com presença de objetos e concentrar a amostragem onde há dados relevantes.

Ao clicar em **`▶ Executar Extração de Frames`**, as imagens são geradas na pasta `frames/raw/images/` da campanha.

---

### Etapa 3: Pré-Anotação e Geração do `import_tasks.json`
Prepara o arquivo de tarefas que será enviado ao Label Studio:

- **Pular Pré-Anotação (Recomendado para dados inéditos)**:
  - Gera as tarefas 100% limpas para rotulação manual humana.
- **Usar Modelo de Detecção**:
  - Seleciona um modelo em `models/` para pré-rotular os frames com bboxes sugeridas.

Ao clicar em **`▶ Gerar import_tasks.json`**, o arquivo de tarefas é salvo na campanha.

---

### Etapa 4: Label Studio, ML Backend e Exportação
Fornece integração com a rotulação e monitoramento automático de conclusão:

1. **Servidor de Detecção (ML Backend)**:
   - Opção para ativar o servidor de inferência em segundo plano na porta `9090` e selecionar qual modelo de `models/` ele deve carregar para auxiliar a rotulação ao vivo no Label Studio.
   - Botão **`🚀 Iniciar Label Studio (+ ML Backend)`**.
2. **Pasta de Exportação Automática (`label_studio/finished_tasks`)**:
   - Instrução visual do caminho exato onde o arquivo JSON exportado pelo Label Studio deve ser salvo:
     `campaigns/<campaign_id>/label_studio/finished_tasks/`
3. **Detecção e Métricas Automáticas**:
   - Assim que o arquivo JSON final é colocado nessa pasta, o Dataset Studio detecta o arquivo automaticamente e atualiza a Etapa 4 para **`✓ Concluído`**.
   - Exibe o **Painel de Métricas**: Total de Imagens, Total de Bboxes, Negativos Confirmados e Contagem por Classe/Vídeo.
   - Exibe o botão **`📦 Seguir para Criar Release &rarr;`**.

---

## 4. Criação e Materialização da Release

Ao clicar em **`📦 Seguir para Criar Release`**, você entra na tela de montagem do dataset (`/release.html?id=release_X&campaign=campanha_Y`):

### 1. Divisão por Vídeo Completo (Sem Vazamento / Data Leakage)
- O sistema lista todos os vídeos da campanha e permite atribuir cada vídeo para **`Train`** (Treinamento) ou **`Val`** (Validação).
- Ao alterar o papel dos vídeos, a **Calculadora em Tempo Real** atualiza instantaneamente a contagem de frames e anotações que vão para Treino e para Validação.

### 2. Materialização
- Ao clicar em **`🔨 Materializar Dataset`**, o sistema gera o arquivo `data.yaml`, `manifest.csv` e organiza as imagens e labels no formato exigido pela Ultralytics YOLO.

---

## 5. Treinamento do Modelo YOLO

Na tela da Release materializada:

1. **Seleção de Modelo e Parâmetros**:
   - Escolha o modelo de partida (um novo modelo base como `yolo26n.pt` / `yolov8n.pt` ou um modelo existente em `models/`).
   - Configure o número de **épocas**, **tamanho da imagem (imgsz)**, **batch size** e **dispositivo (CPU/GPU)**.
2. **Execução em Segundo Plano**:
   - Ao clicar em **`🚀 Iniciar Treinamento`**, o treino é disparado via `JobManager` e os resultados ficam salvos em `runs/detect/<release_id>/`.
3. **Terminal em Tempo Real**:
   - Um terminal interativo na tela exibe as métricas de perda, época atual e andamento ao vivo.
4. **Conclusão**:
   - Ao finalizar, o sistema exibe os melhores pesos gerados (`best.pt`) e métricas finais.
