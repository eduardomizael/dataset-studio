# Guia de Resolução de Problemas (Troubleshooting) 🛠️

Este documento reúne soluções práticas para as principais dificuldades e erros que podem surgir ao configurar, executar ou estender o **Dataset Studio**.

---

## 1. Conflito de Portas TCP
O ecossistema do Dataset Studio utiliza até 3 portas de rede simultaneamente:
* `8000`: Servidor principal FastAPI (Painel Web).
* `8080`: Servidor do Label Studio (para anotação).
* `9090`: Servidor de detecção local (ML Backend).

### Sintoma:
O servidor do Dataset Studio ou o Label Studio falha ao iniciar com o erro `Address already in use` ou `Port is busy`.

### Solução:
1. **Identificar o processo que está ocupando a porta**:
   * No Windows (PowerShell):
     ```powershell
     Get-NetTCPConnection -LocalPort 8080 | Format-Table -Property LocalAddress, LocalPort, State, OwningProcess
     ```
   * No Linux / macOS (Terminal):
     ```bash
     lsof -i :8080
     ```
2. **Encerrar o processo culpado**:
   * No Windows:
     ```powershell
     Stop-Process -Id <OwningProcess> -Force
     ```
   * No Linux / macOS:
     ```bash
     kill -9 <PID>
     ```
3. **Mudar a porta do Dataset Studio**:
   * Você pode mudar a porta padrão do Painel Web FastAPI ao inicializar via CLI:
     ```bash
     uv run dataset-studio.py --port 8010
     ```

---

## 2. Processos Órfãos / Zumbis do Label Studio ou ML Backend
Ao fechar o terminal do Dataset Studio abruptamente (usando `Ctrl+C` repetidamente ou fechando a janela do console), os subprocessos do Label Studio ou do ML Backend rodando em segundo plano podem continuar ativos.

### Sintoma:
Você tenta reiniciar o Dataset Studio e o Label Studio, mas as portas `8080` ou `9090` já estão ocupadas por processos invisíveis em segundo plano.

### Solução:
O Dataset Studio gerencia processos filhos de forma segura no encerramento normal. Porém, se houver travamento:
* No Windows, encerre os processos `label-studio.exe` ou processos `python` que estejam rodando o ML Backend através do Gerenciador de Tarefas ou pelo terminal:
  ```powershell
  taskkill /f /im label-studio.exe
  taskkill /f /im python.exe
  ```
* No Linux / macOS:
  ```bash
  pkill -f label-studio
  pkill -f ml_backend
  ```

---

## 3. Problemas com Leitura ou Codecs de Vídeo (OpenCV)
O Dataset Studio depende do OpenCV para inspecionar os arquivos de vídeo e extrair os quadros na pasta da campanha.

### Sintoma:
A extração de frames falha silenciosamente, cria arquivos vazios (`0 bytes`), ou exibe erros como `Assertion failed` ou `VideoReader could not open file`.

### Solução:
* **Verificar o codec do vídeo**: Nem todos os codecs de vídeo são suportados por padrão pelo OpenCV (especialmente se o OpenCV foi instalado sem suporte a codecs proprietários ou FFmpeg). Certifique-se de que os vídeos estão codificados em **H.264 (AVC)** dentro do container **MP4**.
* **Testar no Python**: Tente abrir o vídeo manualmente em um terminal Python interativo para validar o OpenCV:
  ```python
  import cv2
  cap = cv2.VideoCapture("videos/seu_video.mp4")
  print("Vídeo aberto?", cap.isOpened())
  ret, frame = cap.read()
  print("Frame lido?", ret)
  cap.release()
  ```
  Se retornar `False`, re-codifique o vídeo utilizando ferramentas como o HandBrake ou FFmpeg:
  ```bash
  ffmpeg -i video_original.avi -c:v libx264 -an video_convertido.mp4
  ```

---

## 4. O Dataset Studio não detecta o JSON do Label Studio
Na Etapa 4, o sistema fica travado aguardando o arquivo JSON final e não habilita o botão de seguir para a release.

### Sintoma:
Você colocou o JSON exportado do Label Studio na pasta `label_studio/finished_tasks/`, mas nada acontece ou a interface não atualiza.

### Solução:
1. **Verificar a estrutura de pastas**: O caminho correto para salvar a exportação é:
   `WORKSPACE/campaigns/<campaign_id>/label_studio/finished_tasks/nome_do_arquivo.json`
   Certifique-se de que o ID da campanha na pasta corresponde exatamente ao ID visualizado.
2. **Formato do Arquivo**: Certifique-se de que exportou no formato **JSON** clássico do Label Studio (não JSON-MIN). O parser exige o formato nativo contendo o objeto de tarefas (`tasks`) e suas respectivas anotações (`annotations`).
3. **Erros no JSON**: Abra o arquivo no editor de texto para certificar-se de que o download foi concluído sem corrompimento e que ele não está vazio.

---

## 5. Falta de Bibliotecas ou Erros de CUDA (PyTorch/Ultralytics)
Erros que acontecem durante a extração inteligente de frames, pré-anotação por modelo ou ao rodar o comando de treinamento do YOLO.

### Sintoma:
Mensagens como `ModuleNotFoundError: No module named 'ultralytics'`, `CUDA out of memory` ou `CUDA error: device-side assert triggered`.

### Solução:
* **Dependências Não Instaladas**: Se você estiver executando o comando sem o `uv`, as dependências podem não estar isoladas. Execute sempre com `uv run` ou ative o ambiente virtual (`.venv\Scripts\activate` no Windows).
* **Problemas de CUDA (Memória de Vídeo)**:
  * Se a GPU (VRAM) ficar sem espaço durante o treinamento, diminua o tamanho do lote de treino (`batch size`). O padrão `-1` tenta calcular automaticamente o máximo tolerado, mas você pode definir um valor estático baixo na CLI ou na Web (ex: `batch=4` ou `batch=8`).
  * Se a máquina não possuir GPU dedicada com suporte CUDA, selecione explicitamente o dispositivo como `cpu` nas configurações de treino ou inicialização do ML Backend.
