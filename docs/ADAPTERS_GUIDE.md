# Guia de Adaptadores Customizados - Ports & Adapters 🔌

O **Dataset Studio** é desenhado sob os princípios de **Clean Architecture** (Arquitetura Hexagonal), desacoplando totalmente as regras de negócio de domínio da infraestrutura externa e de frameworks de terceiros.

A comunicação com frameworks de machine learning (como inferência de imagens e execução de treinamentos) ocorre por meio de **Portas (Ports)** e **Adaptadores (Adapters)**. Se você precisar estender a ferramenta para suportar outros modelos ou frameworks de treino além do Ultralytics YOLO, basta implementar um novo adaptador.

---

## 1. O Conceito de Portas no Domínio
As portas definem interfaces formais (contratos) na camada de negócio, enquanto os adaptadores são as implementações concretas na camada periférica.

As duas portas principais do Dataset Studio residem em `src/dataset_studio/ports/`:

1. **`Predictor`** (em `predictor.py`): Contrato para modelos que executam inferência e predição de bounding boxes em imagens.
2. **`Trainer`** (em `trainer.py`): Contrato para montagem de comandos CLI para o loop de treinamento.

---

## 2. A Porta `Predictor`

Qualquer modelo de detecção de objetos de visão computacional que você desejar integrar para a **Extração Inteligente** ou como **ML Backend de sugestão de rótulos** no Label Studio deve aderir ao protocolo `Predictor`.

### Definição do Contrato:
```python
from typing import NamedTuple, Protocol
import numpy as np

class Detection(NamedTuple):
    class_id: int
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]  # Coordenadas em pixel (xmin, ymin, xmax, ymax)

class Predictor(Protocol):
    def predict(self, image: np.ndarray) -> list[Detection]:
        """Recebe uma imagem OpenCV (NumPy array) e retorna a lista de detecções."""
        ...

    @property
    def model_version(self) -> str:
        """Retorna uma string identificadora da versão do modelo."""
        ...
```

### Exemplo: Implementando um Adaptador de Inferência Customizado (ex: PyTorch Hub / SSD)
Imagine que você queira utilizar um modelo SSD do PyTorch Hub em vez do YOLO:

```python
# src/dataset_studio/adapters/custom_ssd/predictor.py

import torch
import numpy as np
from dataset_studio.ports.predictor import Detection, Predictor

class CustomSSDPredictor(Predictor):
    def __init__(self, model_name: str = "ssd300_vgg16_coco", conf: float = 0.25):
        self.model_name = model_name
        self.conf = conf
        self._model = None

    def load(self):
        if self._model is None:
            # Carrega o modelo do PyTorch Hub
            self._model = torch.hub.load("pytorch/vision:v0.10.0", "ssd300_vgg16", pretrained=True)
            self._model.eval()

    def predict(self, image: np.ndarray) -> list[Detection]:
        self.load()
        # 1. Pré-processar a imagem numpy
        # 2. Executar inferência: outputs = self._model(tensor)
        # 3. Converter os outputs para coordenadas absolutas xyxy
        # 4. Filtrar por confiança (self.conf)
        # 5. Retornar lista de Detection
        detections = []
        
        # Código ilustrativo de preenchimento
        # detections.append(Detection(class_id=0, confidence=0.89, bbox_xyxy=(10.0, 20.0, 150.0, 250.0)))
        
        return detections

    @property
    def model_version(self) -> str:
        return f"hub-{self.model_name}-v1"
```

---

## 3. A Porta `Trainer`

Para estender os frameworks de treinamento para além do Ultralytics YOLO (por exemplo, treinar usando RT-DETR, MMDetection ou TensorFlow Object Detection API), você deve implementar a porta `Trainer`.

### Definição do Contrato:
```python
from pathlib import Path
from typing import Protocol
from dataset_studio.ports.trainer import TrainingParams

class Trainer(Protocol):
    def build_command(self, data_yaml_path: Path, params: TrainingParams) -> list[str]:
        """Recebe o caminho do manifesto materializado e os parâmetros configurados pelo usuário
        e retorna uma lista de strings contendo o comando completo de shell para execução.
        """
        ...
```

O `TrainingParams` carrega informações estruturadas de treinamento e um dicionário de argumentos livres (`extra_args`), dando flexibilidade total ao adaptador.

### Exemplo: Implementando um Adaptador de Treino Customizado
```python
# src/dataset_studio/adapters/custom_engine/trainer.py

import sys
from pathlib import Path
from dataset_studio.ports.trainer import Trainer, TrainingParams

class CustomEngineTrainer(Trainer):
    def build_command(self, data_yaml_path: Path, params: TrainingParams) -> list[str]:
        # Monta a linha de comando para chamar o script próprio de treinamento
        cmd = [
            sys.executable,
            "scripts/train_my_custom_model.py",
            f"--config={data_yaml_path.resolve()}",
            f"--epochs={params.epochs}",
            f"--batch-size={params.batch}",
            f"--lr={params.lr0}",
        ]
        
        if params.device:
            cmd.append(f"--device={params.device}")
            
        # Adicionar parâmetros extras livres
        for key, val in params.extra_args.items():
            cmd.append(f"--{key}={val}")
            
        return cmd
```

---

## 4. Onde Configurar os Novos Adaptadores?

Atualmente, os adaptadores padrão são instanciados e chamados nos seguintes locais:

* **Inferência (Amostragem Inteligente / Pré-anotação / ML Backend)**:
  A extração e pré-anotação são orquestradas em `src/dataset_studio/adapters/opencv/media.py`. O runner do backend fica em `src/dataset_studio/adapters/label_studio/runner.py`, e a conversão para o protocolo do Label Studio fica em `ml_backend.py`. O adaptador padrão é `UltralyticsPredictor`.
* **Treinamento**:
  O comando é gerado em `src/dataset_studio/application/version_service.py` na função `training_recipe`, que instancia `UltralyticsCommandTrainer`. O adaptador usa `sys.executable`, portanto executa no ambiente do próprio Dataset Studio.

Se você criar um novo adaptador, basta substituir ou estender essas chamadas de instanciação, injetando sua nova classe concreta. Como a lógica de negócio só conhece os contratos `Predictor` e `Trainer`, nenhuma outra linha do domínio precisará ser alterada.

---

## 5. Requisitos para um Predictor de produção

Além do protocolo mínimo, um adaptador usado pelo fluxo atual deve considerar:

- Carregamento antecipado no processo do ML Backend, para que `/health` só fique disponível depois de o modelo estar utilizável.
- `model_version` estável, preferencialmente derivada de nome, tamanho, data ou hash do artefato.
- Coordenadas `bbox_xyxy` absolutas em pixels.
- Mapeamento de `class_id` compatível com `source.yaml`.
- Opções de inferência como confiança, device, imgsz, IoU, max_det e meia precisão.
- Aplicação de ROI antes da inferência, quando configurada.
- Ausência de caminhos fixos para repositórios ou ambientes externos.

O `GenericLabelStudioBackend` recebe `class_names` e `default_root` explicitamente. Referências `/data/local-files/?d=...` são resolvidas apenas dentro do document root permitido.

## 6. Requisitos para um Trainer de produção

O comando retornado deve:

- ser uma lista de argumentos, sem depender de interpretação de shell;
- usar caminhos absolutos para `data.yaml` e `project`;
- respeitar o `training_id` recebido em `params.name`;
- permitir `exist_ok=True`, pois `workflow_job.json` e `train.log` são criados no diretório antes do Ultralytics iniciar;
- produzir artefatos dentro de `runs/detect/<training_id>/`;
- não escolher executáveis de outro projeto.

O estado persistido do treinamento não pertence ao adaptador. Essa responsabilidade é do `JobManager`, que grava `workflow_job.json` ao enfileirar, iniciar, concluir, falhar ou cancelar a execução.
