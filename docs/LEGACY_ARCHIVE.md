# Arquivo físico de datasets e runs legados

O arquivo legado preserva árvores históricas de datasets ou treinamentos que
não podem ser convertidas honestamente em versões atuais do Dataset Studio.
Cada arquivo é armazenado uma única vez em um object store por SHA-256,
enquanto cada snapshot mantém um manifest imutável com os caminhos originais.

```text
dataset/archive/
├── objects/<prefixo>/<sha256>
└── snapshots/<snapshot_id>/
    ├── manifest.csv
    └── snapshot.yaml
```

Esse mecanismo é indicado quando:

- a origem audiovisual não existe mais;
- os caminhos históricos foram sobrescritos;
- somente o dataset materializado sobreviveu;
- um run antigo contém resultados, gráficos e checkpoints não produzidos pelo
  Dataset Studio;
- várias árvores repetem os mesmos arquivos.

Ele não aumenta artificialmente a confiança da proveniência. Um snapshot
reconstruído continua marcado como provável ou incompleto no registry quando
essa é a qualidade real da evidência.

## Importar

```powershell
uv run --all-extras dataset-studio archive import `
  --id fish-legacy-split `
  --source C:\caminho\fish_detection\dataset\split
```

Snapshots existentes não são sobrescritos. Repetir o comando valida a árvore
de origem contra o manifest já fixado.

## Verificar

Somente o arquivo:

```powershell
uv run --all-extras dataset-studio archive verify --id fish-legacy-split
```

Arquivo e origem:

```powershell
uv run --all-extras dataset-studio archive verify `
  --id fish-legacy-split `
  --source C:\caminho\fish_detection\dataset\split
```

## Reconstruir

```powershell
uv run --all-extras dataset-studio archive materialize `
  --id fish-legacy-split `
  --destination C:\temp\split-restaurado
```

O destino precisa estar vazio. Todos os objetos são verificados antes da
materialização.

## Relação com o registry

Datasets e runs podem registrar `physical_archive` com:

- ID do snapshot;
- subcaminhos correspondentes;
- caminho e SHA-256 do manifest;
- data da vinculação.

O validador do registry confere o manifest arquivado. Os caminhos operacionais
dos artefatos continuam independentes, permitindo manter cópias de acesso
rápido sem perder a referência à preservação integral.

Na migração do `fish_detection`, foram criados:

- `fish-detection-legacy-dataset`: árvore completa de datasets;
- `fish-detection-legacy-runs`: resultados históricos de treinamento e
  experimentos.

Os relatórios auditáveis ficam em
`registry/fish_dataset_archive_report.json` e
`registry/fish_runs_archive_report.json`.
