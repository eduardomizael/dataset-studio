# Manual do Usuário - Dataset Studio 🚀

O **Dataset Studio** é uma ferramenta autônoma projetada para organizar, revisar, materializar datasets de visão computacional e executar o treinamento de modelos YOLO com total controle e isolamento.

---

## 1. Inicializando a Aplicação

Para iniciar o **Dataset Studio** e abrir a interface web no navegador, execute o comando na raiz do repositório `dataset-studio`:

```bash
uv sync --all-extras
uv run --all-extras dataset-studio.py
```

A aplicação abrirá automaticamente no endereço: `http://127.0.0.1:8000/`.

---

## 2. Tela Inicial (Dashboard)

A tela inicial exibe um painel dividido em 3 colunas principais:

1. **📁 Origens**: Conjuntos fixos de vídeos, configurações e tarefas de anotação.
2. **📦 Versões**: Datasets configurados ou materializados.
3. **⚡ Treinamentos**: Lista de treinamentos executados ou em andamento na pasta `runs/detect/`.

No canto superior direito, há o botão **`+ Nova Origem de Dados`** e a indicação da pasta raiz do **Workspace**.

---

## 3. Fluxo de uma Origem (4 Etapas)

Ao clicar em uma origem, a tela `/source.html?id=ID_DA_ORIGEM` apresenta quatro etapas. Rotas com `campaign` continuam existindo somente como aliases de compatibilidade.

### Etapa 1: Seleção dos Vídeos
- Ao criar a campanha no modal inicial, você seleciona um ou múltiplos arquivos de vídeo (`.mp4`, `.avi`, `.mkv`) através da caixa de seleção nativa do sistema operacional.
- Novos uploads são armazenados isoladamente em `videos/<source_id>/`. Nomes inválidos, duplicados e tentativas de sair desse diretório são rejeitados.

---

### Etapa 2: Seleção do Modo de Extração dos Frames
Permite escolher como os frames dos vídeos serão extraídos para imagem:

- **Modo Uniforme (Sem Filtro)**:
  - Extrai 1 frame a cada N quadros definidos pelo usuário (`frame_step`, ex: 30 = 1 frame/segundo a 30fps).
  - Recomendado para amostragem limpa e regular de novos vídeos.
- **Modo Inteligente (Com Modelo)**:
  - Utiliza um modelo pré-treinado localizado na pasta `models/` para detectar regiões com presença de objetos e concentrar a amostragem onde há dados relevantes.

Ao clicar em **`▶ Executar Extração de Frames`**, as imagens são geradas em `dataset/sources/<source_id>/frames/raw/images/`, e a configuração usada fica registrada em `source.yaml` e `frame_manifest.json`.

---

### Etapa 3: Pré-Anotação e Geração do `import_tasks.json`
Prepara o arquivo de tarefas que será enviado ao Label Studio:

- **Pular Pré-Anotação (Recomendado para dados inéditos)**:
  - Gera as tarefas 100% limpas para rotulação manual humana.
- **Usar Modelo de Detecção**:
  - Seleciona um modelo em `models/` para pré-rotular os frames com bboxes sugeridas.

Ao clicar em **`▶ Gerar import_tasks.json`**, o arquivo é salvo em `dataset/sources/<source_id>/label_studio/import_tasks.json`. A partir desse momento, a origem fica fixada: extração, classes e esse arquivo não podem ser reconstruídos no mesmo ID.

---

### Etapa 4: Label Studio, ML Backend e Exportação
Fornece integração com a rotulação e monitoramento automático de conclusão:

1. **Configuração única da integração**:
   - Na primeira utilização neste computador, inicie o Label Studio, abra `Account & Settings > Access Token` e cole o token no painel **Integração automática**.
   - O token é validado pela API oficial e armazenado fora do repositório, no perfil local do usuário. Ele não precisa ser informado novamente para cada origem ou versão de dataset.
   - Nas utilizações seguintes, o Dataset Studio cria ou reconhece o projeto, aplica `labeling_config.xml`, importa as tarefas uma única vez e configura a fila automaticamente.
2. **Escolha segura das predições**:
   - A ferramenta seleciona uma versão que cubra todas as tarefas, habilita as preanotações na fila, usa ordem sequencial e limita cada tarefa a uma anotação.
   - Se nenhuma versão cobrir tudo, a interface informa a quantidade exata e exige confirmação antes de permitir imagens sem caixas.
   - Se a origem foi criada para rotulação manual, a ausência de predições é tratada como intencional e não gera alerta.
3. **Servidor de Detecção (ML Backend)**:
   - Opção para ativar o servidor de inferência em segundo plano na porta `9090` e selecionar qual modelo de `models/` ele deve carregar para auxiliar a rotulação ao vivo no Label Studio.
   - Quando habilitado, o backend é registrado ou atualizado automaticamente no projeto pela API; não é necessário abrir as configurações de Machine Learning do Label Studio.
   - Botão **`🚀 Iniciar Label Studio (+ ML Backend)`**.
4. **Pasta de Exportação Automática (`label_studio/finished_tasks`)**:
   - Instrução visual do caminho exato onde o arquivo JSON exportado pelo Label Studio deve ser salvo:
     `dataset/sources/<source_id>/label_studio/finished_tasks/`
5. **Detecção e Métricas Automáticas**:
   - O Dataset Studio detecta os JSONs dessa pasta, mas não os aceita silenciosamente. O usuário escolhe explicitamente qual exportação transformar em revisão.
   - Exibe o **Painel de Métricas**: Total de Imagens, Total de Bboxes, Negativos Confirmados e Contagem por Classe/Vídeo.
   - Cada exportação pode gerar uma revisão independente. A mesma origem admite múltiplas revisões e múltiplas versões de dataset.

O vínculo fica registrado em `label_studio/integration.json`, com o ID do projeto, hashes dos arquivos fixados, versão de predição escolhida e cobertura. O token de acesso nunca é gravado nesse arquivo.

---

## 4. Criação e Materialização da Versão

Ao escolher uma revisão, você entra na montagem da versão (`/version.html`). `release` é mantido como alias legado.

### 1. Divisão por Vídeo Completo (Sem Vazamento / Data Leakage)
- O sistema exige que cada vídeo seja atribuído exatamente uma vez a `train`, `val`, `test_normal` ou `test_stress`.
- A divisão ocorre por vídeo completo, evitando que frames correlacionados vazem entre splits.

### 2. Materialização
- Ao clicar em **`🔨 Materializar Dataset`**, o sistema constrói tudo em staging e só publica após sucesso integral. São gerados `data.yaml`, `data_test_stress.yaml` quando aplicável, `manifest.csv`, `build_report.json`, imagens e labels YOLO.
- Uma versão materializada não pode ser reconstruída ou editada no mesmo ID. Para mudar revisão ou splits, crie outra versão.

---

## 5. Treinamento do Modelo YOLO

Na tela da versão materializada:

1. **Seleção de Modelo e Parâmetros**:
   - Escolha o modelo de partida (um novo modelo base como `yolo26n.pt` / `yolov8n.pt` ou um modelo existente em `models/`).
   - Configure o número de **épocas**, **tamanho da imagem (imgsz)**, **batch size** e **dispositivo (CPU/GPU)**.
2. **Execução em Segundo Plano**:
   - Cada clique cria um ID único no formato `t_<timestamp>_<sufixo>` e salva os resultados em `runs/detect/<training_id>/`.
   - O `workflow_job.json` persiste a associação entre treinamento e versão, parâmetros, comando e estado.
3. **Terminal em Tempo Real**:
   - Um terminal interativo na tela exibe as métricas de perda, época atual e andamento ao vivo.
4. **Conclusão**:
   - Ao finalizar, o sistema exibe os melhores pesos gerados (`best.pt`) e métricas finais.
   - O registry consolida automaticamente o dataset, modelo inicial, modelo-pai,
     hashes, melhor época e checkpoint resultante.
   - Pesos promovidos podem continuar com nomes amigáveis, mas sua identidade é
     o `model_id` associado ao SHA-256.
   - Ao promover pela interface, a ferramenta também cria automaticamente
     `deployments/<model_id>/deployment_manifest.yaml` e uma cópia imutável do
     peso. Esse diretório pode ser copiado para a aplicação ou equipamento de
     inferência sem depender do servidor do Dataset Studio.

---

## 6. Exclusão consciente

Origens, revisões, versões e treinamentos podem ser excluídos. A ferramenta informa o impacto, mas não substitui a decisão do usuário:

- A prévia mostra versões e treinamentos dependentes.
- Para origens, também informa vídeos compartilhados com outras origens.
- O usuário escolhe exclusão em cascata ou preservação dos dependentes, mesmo que fiquem inválidos.
- Ao excluir uma origem, é possível apagar ou preservar os vídeos físicos.
- A confirmação exige digitar exatamente o ID do recurso.

Exclusão é diferente de mutação: um recurso existente continua imutável durante seu uso, mas pode ser removido explicitamente pelo usuário.

---

## 7. Modelo de ciclo de vida

1. Uma origem contém vídeos e configuração de extração.
2. `import_tasks.json` fixa a origem.
3. Exportações do Label Studio geram revisões independentes.
4. Uma versão escolhe revisões e quatro splits por vídeo.
5. A materialização fixa fisicamente aquela versão.
6. A mesma versão materializada pode alimentar quantos treinamentos forem necessários.
