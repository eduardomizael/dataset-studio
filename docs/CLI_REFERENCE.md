# Referência da CLI — Dataset Studio

Instale e execute a CLI no ambiente completo do repositório:

~~~powershell
uv sync --all-extras
uv run --all-extras dataset-studio --workspace C:\meu_workspace <comando>
~~~

O workspace padrão é o diretório atual. source e version são os termos canônicos; campaign e release são aliases legados.

## Origens

### source create

Cria source.yaml a partir de vídeos já existentes.

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
uv run --all-extras dataset-studio version create --id dataset_v1 --sources origem_peixes --assignments-json '{"train":["origem_peixes/video_01.mp4"],"val":["origem_peixes/video_02.mp4"],"test_normal":[],"test_stress":[]}'
~~~

Observações:

- Todos os vídeos precisam aparecer exatamente uma vez.
- train e val são obrigatórios.
- A CLI seleciona a revisão disponível conforme as regras atuais do domínio. Para escolher explicitamente uma revisão por origem, use a interface ou a API.

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

## Limitações atuais da CLI

A CLI ainda não expõe:

- upload multipart isolado;
- escolha completa dos parâmetros da extração inteligente;
- início do Label Studio e ML Backend;
- prévia de impacto e exclusões;
- escolha explícita de annotation_revisions no version create.

Use a API REST ou o painel web para essas operações. Não edite YAMLs ou artefatos fixados manualmente.
