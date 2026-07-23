# Catálogo de linhagem

Este diretório é um **índice global derivado**. Ele permite consultar relações
que atravessam recursos — origem → versão → treinamento → modelo — sem mover os
manifestos para fora de seus próprios diretórios.

As fontes canônicas continuam sendo:

- `dataset/sources/<source_id>/source.yaml`;
- `dataset/versions/<version_id>/version.yaml`;
- `runs/detect/<training_id>/run.yaml`;
- `deployments/<deployment_id>/deployment_manifest.yaml`.

Os registros em `registry/sources/`, `registry/datasets/` e `registry/runs/`
apontam para esses manifestos e guardam seus hashes. `models.yaml` reúne
identidades lógicas e aliases porque um mesmo peso pode aparecer em mais de um
caminho.

O catálogo pode ser reconstruído para recursos nativos com:

```powershell
python -m dataset_studio.utils.sync_lineage_catalog --workspace .
```

Alterar manualmente um manifesto canônico sem sincronizar o catálogo é
detectado por `dataset-studio --workspace . registry status`.
