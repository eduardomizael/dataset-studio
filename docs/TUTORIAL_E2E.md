# Tutorial Passo a Passo End-to-End: Detecção de Peixes 🐟

Este tutorial prático guiará você por todas as etapas de uso do **Dataset Studio** a partir do zero. Utilizaremos o caso de estudo de **contagem de peixes em uma canaleta física** para exemplificar o ciclo completo: desde a importação de vídeos inéditos até o treinamento do modelo YOLO sem vazamento de dados.

---

## Pré-requisitos
* Ter o `uv` instalado no sistema.
* Ter pelo menos um arquivo de vídeo de teste (ex: `peixes_canaleta_01.mp4`).
* Ter um modelo pré-treinado na pasta `models/` (ex: `models/yolov8n.pt`), caso queira testar a extração inteligente ou o ML Backend.

---

## Passo 1: Inicializando o Painel
Abra o terminal na raiz do projeto `dataset-studio` e inicialize o servidor local:

```bash
uv run dataset-studio.py
```

O navegador abrirá automaticamente em `http://127.0.0.1:8000/`.

---

## Passo 2: Criando a Origem de Dados (Campanha)
1. No canto superior direito, clique em **`+ Nova Origem de Dados`** (ou **`+ Nova Campanha`**).
2. Preencha os campos no modal:
   * **ID da Origem**: `campanha_peixes`
   * **Seleção de Vídeos**: Clique na caixa tracejada e selecione os vídeos locais (ex: `peixes_canaleta_01.mp4` e `peixes_canaleta_02.mp4`).
   * **Classes de Objetos**: Escreva `peixe` (se forem múltiplas, separe por vírgula).
3. Clique em **`Criar Origem de Dados`**. Os vídeos serão salvos internamente no diretório `videos/` e a campanha aparecerá na primeira coluna do Dashboard.

---

## Passo 3: Amostragem e Extração de Frames
Clique na campanha recém-criada na tela inicial para entrar nos detalhes dela:

1. A **Etapa 1 (Seleção dos Vídeos)** já estará concluída.
2. Abra a **Etapa 2 (Seleção do Modo de Extração)**:
   * **Modo Uniforme (Recomendado)**: Selecione-o e defina `30` como intervalo (`frame_step`). Isso extrairá 1 frame a cada 30 quadros do vídeo (aproximadamente 1 imagem por segundo).
   * **Modo Inteligente**: Selecione-o se tiver um modelo base em `models/` e desejar extrair apenas frames onde o modelo detectar a presença de peixes.
3. Clique em **`▶ Executar Extração de Frames`**. Uma barra de progresso ou logs serão exibidos, salvando as imagens extraídas na pasta `campaigns/campanha_peixes/frames/raw/images/`.

---

## Passo 4: Pré-Anotação e Carga no Label Studio
1. Abra a **Etapa 3 (Pré-Anotação)**:
   * Para este tutorial, selecione **`Pular Pré-Anotação`** para realizar a rotulação de forma manual e limpa.
   * Se preferir usar predições de um modelo prévio como ponto de partida (sugestão de caixas), selecione **`Usar Modelo de Detecção para Pré-Anotar`** e selecione o arquivo correspondente em `models/`.
2. Clique em **`▶ Gerar import_tasks.json`**. O arquivo estruturado de tarefas será salvo na pasta da campanha.

---

## Passo 5: Rotulação no Label Studio (Human-in-the-Loop)
1. Abra a **Etapa 4**.
2. Opcional: Marque a caixa **`Iniciar Servidor de Detecção em Segundo Plano (ML Backend na porta 9090)`** e escolha um modelo para auxiliar a rotulação ao vivo.
3. Clique em **`🚀 Iniciar Label Studio (+ ML Backend)`**. O Dataset Studio abrirá uma aba no navegador com a interface do Label Studio rodando em `http://127.0.0.1:8080`.
4. **No Label Studio**:
   * Crie um projeto de detecção de objetos (Object Detection).
   * Vá em **Settings > Labeling Interface** e defina o painel XML correspondente à sua classe `peixe`.
   * Vá em **Settings > Cloud Storage**, adicione um armazenamento do tipo **Local Files** apontando para o diretório físico absoluto da campanha e faça o Sync.
   * Importe o arquivo `import_tasks.json` gerado no Passo 4.
   * Realize as anotações (desenhe as bboxes nos peixes).
   * Ao finalizar, clique em **`Export`**, selecione o formato **JSON** e baixe o arquivo.

---

## Passo 6: Conclusão da Campanha e Revisão
1. Salve o arquivo JSON exportado do Label Studio dentro da pasta de destino da campanha:
   `campaigns/campanha_peixes/label_studio/finished_tasks/` (você pode renomear o arquivo para algo como `export.json`).
2. O Dataset Studio detectará o arquivo automaticamente em poucos segundos.
3. A Etapa 4 atualizará o status para **`✓ Concluído`** e abrirá o painel de métricas consolidado, mostrando o número exato de peixes anotados.
4. Clique no botão **`📦 Seguir para Criar Release`**.

---

## Passo 7: Divisão de Splits por Vídeo Completo (Sem Data Leakage)
Na tela de criação da release (`release.html`):

1. O sistema listará os dois vídeos originais (`peixes_canaleta_01.mp4` e `peixes_canaleta_02.mp4`).
2. **Evite o Vazamento Temporal**: Em datasets de vídeo, frames consecutivos de um mesmo vídeo são altamente correlacionados. Se você misturar frames do mesmo vídeo em Treino e Validação, seu modelo terá um mAP artificialmente alto durante o treino, mas falhará em novos vídeos.
3. Defina a atribuição:
   * `peixes_canaleta_01.mp4` &rarr; Atribua para **Train** (Treino).
   * `peixes_canaleta_02.mp4` &rarr; Atribua para **Val** (Validação).
4. Observe a calculadora na tela recalculando em tempo real o balanceamento de frames e de caixas em cada split.
5. Defina o ID da release como `release_peixes_v1`.
6. Clique em **`🔨 Materializar Dataset`**. O sistema estruturará os subdiretórios `train/images`, `train/labels`, `val/images` e `val/labels` no formato padrão YOLO e gerará o arquivo `data.yaml`.

---

## Passo 8: Executando o Treinamento YOLO
1. Na mesma tela (agora com o dataset materializado):
   * Escolha o modelo base (ex: `yolo26n.pt` ou outro modelo em `models/`).
   * Defina `50` épocas e tamanho de imagem `640`.
2. Clique em **`🚀 Iniciar Treinamento`**.
3. O painel ativará o monitor em tempo real (terminal de logs do YOLO) na tela. Você poderá acompanhar as épocas passando e a perda (loss) decaindo.
4. Ao final do treino, os melhores pesos resultantes (`best.pt`) estarão salvos no diretório:
   `runs/detect/release_peixes_v1/weights/best.pt`.
5. O novo modelo estará pronto para ser testado e implantado no seu sistema de borda!
