# Referência da API REST — Dataset Studio

A API FastAPI é local por padrão em http://127.0.0.1:8000. O prefixo canônico é /api.

Os termos source e version são canônicos. As rotas campaign e release são aliases de compatibilidade e podem ser removidas em uma futura versão principal.

## Convenções

- Sucesso: HTTP 200 com JSON.
- Erro de regra de negócio ou confirmação: HTTP 400 com campo detail.
- Recurso inexistente: HTTP 404 nas rotas que fazem essa distinção.
- IDs aceitam letras, números, hífen e sublinhado.
- Exclusões exigem o parâmetro confirm igual ao ID do recurso.

## Workspace e modelos

### GET /api/workspace

Retorna root, sources_root, versions_root, videos_root e os aliases legados.

### GET /api/models

Lista caminhos relativos de modelos .pt dentro de models/.

Exemplo:

~~~json
[
  "models/yolo26n.pt",
  "models/modelo_especializado.pt"
]
~~~

## Origens

Todas as rotas abaixo também aceitam /api/campaigns no lugar de /api/sources, exceto quando indicado.

### GET /api/sources

Lista os IDs existentes.

### POST /api/sources

Cria uma origem a partir de vídeos já presentes no disco.

~~~json
{
  "source_id": "origem_01",
  "videos_dir": "videos/origem_01",
  "video_pattern": "*.mp4",
  "video_files": ["video_01.mp4"],
  "video_notes": {"video_01.mp4": "iluminação normal"},
  "classes": ["peixe"]
}
~~~

### POST /api/sources/upload

Recebe multipart/form-data:

- source_id ou campaign_id;
- classes como lista JSON serializada;
- video_notes como objeto JSON serializado;
- um ou mais campos videos.

Os arquivos passam por staging, validação de nome e isolamento em videos/<source_id>/. Uploads duplicados ou com path traversal são rejeitados.

### GET /api/sources/{source_id}

Retorna vídeos, detalhes das mídias, extração configurada, backend de anotação, frames, tasks, revisões, exportações encontradas e next_action.

Esta consulta não aceita exportações e não cria revisões.

### POST /api/sources/{source_id}/extract

Configura e executa a extração. Só pode ser chamada antes de existirem frames.

Modo uniforme:

~~~json
{
  "mode": "uniform",
  "uniform_frame_step": 30
}
~~~

Modo inteligente:

~~~json
{
  "mode": "smart",
  "model": "models/modelo.pt",
  "confidence": 0.25,
  "scan_step": 15,
  "dense_step": 30,
  "sparse_step": 90,
  "margin": 45,
  "max_negatives_per_video": 15
}
~~~

O modelo precisa estar dentro de models/.

### POST /api/sources/{source_id}/import-tasks

Gera label_studio/import_tasks.json. Depois do sucesso, a origem fica fixada.

Sem pré-anotação:

~~~json
{"mode": "none"}
~~~

Com pré-anotação:

~~~json
{
  "mode": "model",
  "model": "models/modelo.pt",
  "confidence": 0.25
}
~~~

O valor existing preserva predições já existentes no frame_manifest.json.

### GET /api/sources/{source_id}/finished-tasks

Inspeciona os JSONs em label_studio/finished_tasks/ e retorna métricas, erros e validade. Não cria revisões.

### POST /api/sources/{source_id}/accept-export

Cria uma revisão explícita e imutável.

~~~json
{
  "path": "C:\\workspace\\dataset\\sources\\origem_01\\label_studio\\finished_tasks\\export.json",
  "revision_id": "rev_001",
  "allow_pending": false
}
~~~

allow_pending=true permite snapshot parcial e marca a versão derivada como provisória.

### POST /api/sources/{source_id}/start-label-studio

~~~json
{
  "enable_ml": true,
  "model": "models/modelo.pt",
  "allow_partial_predictions": false
}
~~~

Quando enable_ml é verdadeiro, o ML Backend é iniciado primeiro e precisa responder com saúde UP na porta 9090. Em seguida, o Label Studio é iniciado na porta 8080. A API só retorna online=true quando o Label Studio responde.

Se a credencial única estiver configurada, a rota também cria ou reconhece o projeto da origem, evita reimportação, aplica as configurações seguras, conecta o ML Backend quando solicitado e retorna a URL direta do projeto.

### GET /api/label-studio/settings

Informa se a credencial única está configurada. O token nunca é devolvido.

### POST /api/label-studio/settings

Valida e salva a conexão no perfil local do usuário:

~~~json
{
  "base_url": "http://127.0.0.1:8080",
  "api_key": "token-copiado-do-label-studio"
}
~~~

A autenticação detecta automaticamente token legado ou Personal Access Token.

### DELETE /api/label-studio/settings

Remove a credencial local. Não exclui projetos nem anotações.

### GET /api/sources/{source_id}/label-studio

Retorna o estado da credencial, o vínculo persistido, as versões de predição, a cobertura e uma orientação contextual quando há uma ação manual.

### POST /api/sources/{source_id}/label-studio/prepare

Cria, reconhece ou revalida o projeto de forma idempotente:

~~~json
{
  "allow_partial_predictions": false
}
~~~

Por padrão, predições parciais são recusadas. `allow_partial_predictions=true` representa uma confirmação consciente de que algumas tarefas poderão abrir sem caixas.

## Versões

As rotas também aceitam /api/releases como alias.

### GET /api/versions

Lista versões configuradas e materializadas.

### POST /api/versions/preview-split

Calcula vídeos, frames e caixas por split antes da criação.

~~~json
{
  "source_id": "origem_01",
  "revision_id": "rev_001",
  "assignments": {
    "train": ["origem_01/video_01.mp4"],
    "val": ["origem_01/video_02.mp4"],
    "test_normal": [],
    "test_stress": []
  }
}
~~~

### POST /api/versions

~~~json
{
  "version_id": "dataset_v1",
  "sources": ["origem_01"],
  "annotation_revisions": {"origem_01": "rev_001"},
  "assignments": {
    "train": ["origem_01/video_01.mp4"],
    "val": ["origem_01/video_02.mp4"],
    "test_normal": ["origem_01/video_03.mp4"],
    "test_stress": ["origem_01/video_04.mp4"]
  }
}
~~~

Cada vídeo deve aparecer exatamente uma vez. train e val precisam conter frames utilizáveis.

### POST /api/versions/{version_id}/build

Materializa em staging e publica de forma transacional em dataset/versions/<version_id>/. Uma versão que já possui manifest.csv não pode ser reconstruída.

### GET /api/versions/{version_id}

Retorna configuração, revisões, splits, estado de materialização, build_report e receita de treinamento.

### POST /api/versions/{version_id}/start-train

~~~json
{
  "model": "models/modelo.pt",
  "epochs": 50,
  "imgsz": 640,
  "batch": -1,
  "workers": 0,
  "device": "auto",
  "patience": 50,
  "lr0": 0.01,
  "optimizer": "auto"
}
~~~

Cria um training_id único, enfileira a execução e persiste workflow_job.json em runs/detect/<training_id>/.

## Treinamentos

### GET /api/trainings

Lista os diretórios de runs/detect e seus estados persistidos.

### GET /api/trainings/{training_id}

Retorna parâmetros, logs, métricas, pesos, duração e versão associada. Treinamentos legados tentam recuperar a versão pelo caminho data do args.yaml.

### POST /api/trainings/{training_id}/promote

Promove `best.pt` para `models/` e registra o novo alias no registry. O corpo
aceita `target_name` e `overwrite`. Sobrescrita de um peso diferente é recusada
por padrão; componentes de diretório não são aceitos.

Além da promoção, cria automaticamente um bundle imutável em
`deployments/<model_id>/`.

### POST /api/models/{model_id}/deploy

Exporta um modelo já registrado para um bundle de implantação. Corpo opcional:

```json
{
  "deployment_id": "peixes-producao",
  "artifact_path": "models/modelo.pt"
}
```

O `deployment_id` é imutável. Uma repetição para o mesmo modelo é idempotente;
um identificador já usado por outro modelo é recusado.

### GET /api/registry/status

Valida referências e SHA-256 de datasets, runs, modelos, aliases e artefatos.

### GET /api/registry/models

Retorna o catálogo de modelos e o mapa de aliases físicos.

### DELETE /api/trainings/{training_id}?confirm={training_id}

Apaga logs, métricas e pesos daquele treinamento.

## Exclusão e impacto

### GET /api/deletion-impact/{resource_type}/{resource_id}

resource_type aceita source, revision, version ou training. Para revision, envie também source_id como query parameter.

Retorna dependent_versions, dependent_trainings, shared_video_references e aviso de rastreabilidade.

### DELETE /api/sources/{source_id}

Query parameters:

- confirm: deve ser igual ao source_id;
- cascade: se verdadeiro, remove versões e treinamentos dependentes;
- delete_videos: se verdadeiro, remove também os vídeos associados.

Sem cascata, a exclusão é permitida e pode deixar recursos inválidos.

### DELETE /api/sources/{source_id}/revisions/{revision_id}

Exige confirm e aceita cascade. Sem cascata, versões que apontavam para a revisão podem ficar inválidas.

### DELETE /api/versions/{version_id}

Exige confirm e aceita cascade para treinamentos dependentes.

## Jobs

- GET /api/jobs
- GET /api/jobs/{job_id}
- POST /api/jobs/{job_id}/stop
- POST /api/jobs/{job_id}/cancel

stop interrompe jobs ativos. cancel só se aplica a treinamento ainda enfileirado.
