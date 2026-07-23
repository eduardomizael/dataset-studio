# Catálogo de linhagem de origens, datasets, treinamentos e modelos

O registry é o índice global derivado de proveniência do Dataset Studio. Ele
não substitui os manifestos que vivem junto de cada recurso; relaciona origens,
versões de dataset, execuções, checkpoints, aliases e estados por
identificadores estáveis e SHA-256.

As fontes canônicas são `source.yaml`, `version.yaml`, `run.yaml` e
`deployment_manifest.yaml`. O catálogo existe separado porque relações como
modelo-pai, aliases e uso de um dataset por vários treinamentos atravessam
diretórios. Cada entrada derivada aponta de volta para o manifesto canônico.

## Estrutura

```text
registry/
├── README.md
├── models.yaml
├── aliases.yaml
├── migration_report.json
├── datasets/
│   └── <dataset_id>.yaml
├── sources/
│   └── <source_id>.yaml
└── runs/
    └── <training_id>.yaml
```

Os diretórios `runs/detect/<training_id>/` também recebem:

- `run.yaml`: cópia consultável do registro da execução;
- `workflow_job.json`: estado do processo e associação ao dataset;
- `provenance/`: snapshots de `version.yaml`, `data.yaml`, `manifest.csv` e
  `build_report.json` para novos treinamentos.

## Evidência retroativa e evidência nativa

Todo registro possui:

```yaml
provenance:
  origin: reconstructed  # ou generated
  confidence: confirmed  # confirmed | probable | incomplete
  reconstructed_at: 2026-07-23T...
  evidence:
    - caminho/do/artefato
```

- `generated`: o Dataset Studio registrou a informação durante a operação.
- `reconstructed`: a informação foi recuperada posteriormente.
- `confirmed`: há hash, manifest, `args.yaml`, `results.csv` ou checkpoint.
- `probable`: evidências convergem, mas falta o snapshot exato.
- `incomplete`: uma parte essencial não está mais disponível.

O nível de confiança nunca é elevado automaticamente pela simples cópia de um
arquivo.

## Identidade de modelo

O `model_id` é a identidade lógica. Nomes físicos diferentes podem ser aliases
do mesmo checkpoint quando possuem o mesmo SHA-256.

```yaml
model_id: model-y26n-d03fixed-s43-best
sha256: 2ba345...
parent_model_id: model-y26n-generic
source_run_id: stat_fixed_valtest_yolo26n_base_e150_b16_seed_43
state: baseline
paths:
  - models/yolo26n_fixed_valtest_seed43_best.pt
```

Estados suportados:

- `external` e `base`;
- `experimental` e `baseline`;
- `candidate` e `promoted`;
- `discarded` e `legacy`.

Promoção adiciona um alias e altera o estado. Ela não cria artificialmente um
novo modelo quando o hash do peso é o mesmo.

## Bundle de implantação

Ao promover um treinamento pela interface, o Dataset Studio também cria um
bundle imutável em:

```text
deployments/<model_id>/
├── model.pt
└── deployment_manifest.yaml
```

O manifest relaciona o artefato ao `model_id`, dataset, treinamento,
checkpoint-pai, métricas e parâmetros recuperáveis. O SHA-256 é recalculado
depois da cópia. Aplicações consumidoras devem validar o manifest antes de
carregar o modelo.

Modelos já registrados também podem ser exportados pela CLI:

```powershell
dataset-studio --workspace . registry deploy `
  --model-id model-y26n-d03fixed-s43-best
```

Ou pela API:

```text
POST /api/models/{model_id}/deploy
```

Exportar um modelo candidato, legado ou com proveniência incompleta é
permitido. O bundle registra avisos para que a decisão continue explícita, sem
impedir o usuário.

## Geração automática

Antes de um novo treinamento, o Dataset Studio:

1. registra o hash do modelo inicial;
2. registra ou atualiza o manifest da versão materializada;
3. copia os quatro arquivos de evidência da versão para o run;
4. cria o registro do treinamento com estado `queued`.

Ao terminar, consolida:

- status e hiperparâmetros;
- `results.csv`, métricas finais e melhor época;
- hashes de `best.pt` e `last.pt`;
- relação com o modelo-pai;
- modelo resultante e seus aliases.
- avaliações independentes do `best.pt` em `test_normal` e `test_stress`;
- queda absoluta e relativa das métricas sob estresse.

O teste de estresse ocorre somente depois do treino e nunca seleciona época ou
checkpoint. Falha em uma avaliação é registrada sem apagar um treinamento que
terminou corretamente.

Origens e versões nativas existentes podem ser sincronizadas de forma
idempotente:

```powershell
python -m dataset_studio.utils.sync_lineage_catalog --workspace .
```

## Migração do fish_detection

A migração executada em 23 de julho de 2026 é reproduzível e aditiva:

```powershell
uv run --frozen python -m dataset_studio.utils.fish_history_migration `
  --workspace C:\Users\eduar\Desktop\dataset-studio `
  --source C:\Users\eduar\Desktop\fish_detection
```

Ela:

- copia runs e modelos sem apagar os originais;
- recusa colisões de arquivo com SHA-256 diferente;
- reconhece aliases byte a byte;
- cria manifests D0 a D4 com o nível de confiança apropriado;
- registra os runs que já existiam no Dataset Studio;
- pode ser repetida sem duplicar os artefatos.

O relatório fica em `registry/migration_report.json`.

## Validação

Pela CLI:

```powershell
dataset-studio --workspace . registry status
```

Pela API:

```text
GET /api/registry/status
GET /api/registry/models
GET /api/registry/sources
```

O validador confere:

- pais, datasets e modelos referenciados;
- origens e hashes de seus `source.yaml`;
- hashes de modelos e aliases;
- hashes dos artefatos dos runs;
- hash do manifest do dataset quando disponível;
- arquivos ausentes.

Um aviso não transforma uma reconstrução incompleta em erro. Por exemplo, o
peso genérico YOLO11m usado em `train6` não foi preservado e permanece
explicitamente marcado como ausente.
