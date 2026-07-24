# Referência da CLI — Dataset Studio

Instale e execute a CLI no ambiente completo do repositório:

~~~powershell
uv sync --all-extras
uv run --all-extras dataset-studio --workspace C:\meu_workspace <comando>
~~~

O workspace padrão é o diretório atual. source e version são os termos canônicos; campaign e release são aliases legados.

## Origens

### source create

Cria source.yaml a partir de vídeos já existentes. Segmentos virtuais podem ser
informados com `--capture-units-json`.

~~~powershell
uv run --all-extras dataset-studio source create --id origem_peixes --videos-dir C:\dados\videos --pattern "*.mp4" --classes peixe
~~~

Argumentos:

- --id: identificador único;
- --videos-dir: diretório dos vídeos;
- --pattern: glob usado para selecionar vídeos;
- --classes: uma ou mais classes.

A criação pela CLI usa a configuração uniforme padrão. Para upload isolado e configuração visual de extração inteligente, use a interface web.

### source list

~~~powershell
uv run --all-extras dataset-studio source list
~~~

### source status

~~~powershell
uv run --all-extras dataset-studio source status --id origem_peixes
~~~

## Label Studio e revisões

### build-import

Gera import_tasks.json usando o frame_manifest.json existente.

~~~powershell
uv run --all-extras dataset-studio build-import --source origem_peixes
~~~

Esse comando fixa a origem. Se o arquivo já existir, a operação falha em vez de sobrescrevê-lo.

### accept-revision

~~~powershell
uv run --all-extras dataset-studio accept-revision --source origem_peixes --export C:\exportacoes\project-1.json --revision-id rev_001
~~~

Use --allow-pending somente quando quiser aceitar tarefas adiadas ou incompletas como snapshot provisório.

## Versões

### version create

~~~powershell
uv run --all-extras dataset-studio version create --id dataset_v1 --sources origem_peixes --evaluation-level standard --assignments-json '{"train":["origem_peixes/leva_01"],"val":["origem_peixes/leva_02"],"test_normal":["origem_peixes/leva_03"],"test_stress":[]}'
~~~

Observações:

- Todas as unidades precisam aparecer exatamente uma vez.
- Os splits obrigatórios dependem de `--evaluation-level`.
- `--annotation-revisions-json` escolhe uma revisão por origem.
- `--class-mapping-json` recebe `origem -> classe original -> classe final`.
  Use `null` para ignorar uma classe.
- `--final-classes` define a ordem final dos IDs.
- Alterações semânticas exigem `--acknowledge-class-mapping`.

Exemplo com fusão e descarte:

~~~powershell
uv run --all-extras dataset-studio version create `
  --id dataset_combinado `
  --sources origem_a origem_b `
  --evaluation-level pilot `
  --annotation-revisions-json '{"origem_a":"rev_01","origem_b":"rev_02"}' `
  --assignments-json '{"train":["origem_a/leva_01","origem_b/leva_02"],"val":[],"test_normal":[],"test_stress":[]}' `
  --class-mapping-json '{"origem_a":{"peixe":"peixe"},"origem_b":{"fish":"peixe","bolha":null}}' `
  --final-classes peixe `
  --acknowledge-class-mapping
~~~

### version list

~~~powershell
uv run --all-extras dataset-studio version list
~~~

### version status

~~~powershell
uv run --all-extras dataset-studio version status --id dataset_v1
~~~

### version build

~~~powershell
uv run --all-extras dataset-studio version build --id dataset_v1
~~~

A construção ocorre em staging. Depois de materializada, a versão não pode ser reconstruída no mesmo ID.

### version train

Visualizar a receita:

~~~powershell
uv run --all-extras dataset-studio version train --id dataset_v1 --model models\modelo.pt --epochs 50 --imgsz 640 --device auto --dry-run
~~~

Executar no terminal:

~~~powershell
uv run --all-extras dataset-studio version train --id dataset_v1 --model models\modelo.pt --epochs 50 --imgsz 640 --device auto
~~~

Parâmetros disponíveis:

- --model
- --epochs
- --imgsz
- --batch
- --workers
- --device
- --patience
- --lr0
- --optimizer
- --dry-run

Na CLI, o treinamento é síncrono, mas recebe `training_id` exclusivo e gera os
mesmos registros de proveniência. A fila sequencial, monitoramento em tempo
real e cancelamento de jobs pertencem ao fluxo web/API.

## Registry

Validar datasets, runs, modelos, aliases e hashes:

~~~powershell
uv run --all-extras dataset-studio registry status
~~~

O comando retorna código 1 quando encontra uma inconsistência estrutural ou de
SHA-256. Arquivos históricos sabidamente ausentes aparecem como avisos.

Exportar um modelo registrado para implantação:

~~~powershell
uv run --all-extras dataset-studio registry deploy `
  --model-id model-y26n-d03fixed-s43-best `
  --deployment-id peixes-producao
~~~

O comando cria `deployments/<deployment_id>/deployment_manifest.yaml` e uma
cópia autocontida do artefato. Bundles existentes não são sobrescritos.

## Arquivo físico legado

Importar, validar e reconstruir uma árvore histórica de dataset ou run:

~~~powershell
uv run --all-extras dataset-studio archive import --id legado-d2 --source C:\dados\split
uv run --all-extras dataset-studio archive verify --id legado-d2 --source C:\dados\split
uv run --all-extras dataset-studio archive materialize --id legado-d2 --destination C:\temp\d2
uv run --all-extras dataset-studio archive status
~~~

O arquivo usa SHA-256 para deduplicar conteúdos e nunca sobrescreve um snapshot
existente.

## Limitações atuais da CLI

A CLI ainda não expõe:

- upload multipart isolado;
- escolha completa dos parâmetros da extração inteligente;
- início do Label Studio e ML Backend;
- prévia de impacto e exclusões;

Use a API REST ou o painel web para essas operações. Não edite YAMLs ou artefatos fixados manualmente.
