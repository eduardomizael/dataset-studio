# Tutorial end-to-end — Dataset Studio

Este tutorial percorre o ciclo completo: origem, frames, Label Studio, revisão, versão materializada e treinamentos independentes.

## 1. Preparação

Requisitos:

- uv instalado;
- vídeos locais;
- um modelo .pt em models/ para extração inteligente, pré-anotação ou ML Backend.

Na raiz do repositório:

~~~powershell
uv sync --all-extras
uv run --all-extras dataset-studio.py
~~~

O painel abre em http://127.0.0.1:8000/.

No Windows com NVIDIA, --all-extras instala o extra CUDA 12.8 declarado no projeto. Confira a GPU com:

~~~powershell
uv run --all-extras python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
~~~

## 2. Criar uma origem

1. Clique em Nova Origem de Dados.
2. Informe um ID, por exemplo canaleta_2026_07.
3. Selecione os vídeos.
4. Informe as classes, por exemplo peixe.
5. Opcionalmente, registre uma observação por vídeo.

Os uploads novos ficam em:

    videos/canaleta_2026_07/

O manifesto inicial fica em:

    dataset/sources/canaleta_2026_07/source.yaml

## 3. Extrair frames

Escolha uma das opções:

- Uniforme: extrai um frame a cada uniform_frame_step quadros.
- Inteligente: faz um scan inicial com o modelo, usa dense_step em regiões detectadas e sparse_step para negativos.

No modo inteligente, selecione um modelo dentro de models/. A configuração escolhida é persistida e a interface passa a exibi-la como somente leitura após a extração.

Resultados:

    dataset/sources/canaleta_2026_07/frame_manifest.json
    dataset/sources/canaleta_2026_07/frames/raw/images/

## 4. Gerar import_tasks.json

Escolha:

- Pular pré-anotação: tasks sem sugestões.
- Usar modelo: todas as imagens recebem predições antes da geração.

Depois, clique em Gerar import_tasks.json.

Importante: esse é o ponto de fixação da origem. O mesmo ID não permite reextração nem reconstrução de import_tasks.json. Se a configuração estiver errada, exclua a origem conscientemente e crie outra.

## 5. Iniciar Label Studio

Na etapa de anotação:

1. Opcionalmente habilite o ML Backend e escolha o modelo.
2. Clique em Iniciar Label Studio.
3. O backend de predição sobe em http://127.0.0.1:9090.
4. O Label Studio sobe em http://127.0.0.1:8080.

A API só declara sucesso depois de validar /health do backend e a disponibilidade do Label Studio.

Na primeira utilização neste computador:

1. Entre no Label Studio e copie um token em `Account & Settings > Access Token`.
2. Cole o token no painel Integração automática do Dataset Studio.
3. Clique em Salvar e preparar esta origem.

Essa autorização é feita uma única vez. A partir dela, o Dataset Studio:

- cria ou reconhece o projeto correspondente à origem;
- aplica `label_studio/labeling_config.xml`;
- importa `label_studio/import_tasks.json` apenas quando o projeto está vazio;
- escolhe uma predição com cobertura completa;
- habilita as caixas na fila `Label All Tasks`;
- configura ordem sequencial e uma anotação por tarefa;
- conecta automaticamente o ML Backend quando essa opção está habilitada;
- abre diretamente o projeto correto.

Se houver somente predições parciais, a interface mostra a cobertura antes de continuar. Para uma origem explicitamente manual, nenhuma predição é exigida.

No Label Studio, basta revisar/anotar as imagens e exportar no formato JSON nativo.

Os caminhos em import_tasks.json usam /data/local-files com o workspace como document root. Não é necessário duplicar as imagens.

## 6. Criar revisões

Copie cada exportação para:

    dataset/sources/canaleta_2026_07/label_studio/finished_tasks/

O Dataset Studio inspeciona todos os JSONs e mostra métricas, mas não aceita automaticamente nenhum deles. Escolha explicitamente a exportação desejada.

Cada aceite cria:

    dataset/sources/canaleta_2026_07/label_studio/revisions/<revision_id>/

Você pode aceitar quantas revisões quiser. Uma revisão parcial só deve usar allow_pending quando essa provisoriedade for intencional.

## 7. Criar uma versão

Escolha uma revisão e atribua cada vídeo exatamente uma vez:

- train;
- val;
- test_normal;
- test_stress.

Exemplo:

- vídeos principais de treinamento em train;
- um vídeo representativo em val;
- aquisição comum independente em test_normal;
- iluminação, densidade ou movimento difíceis em test_stress.

Nunca divida frames do mesmo vídeo entre splits.

## 8. Materializar

Clique em Materializar Dataset. A construção acontece em staging e só é publicada após sucesso integral.

Estrutura final:

    dataset/versions/dataset_canaleta_v1/
      version.yaml
      manifest.csv
      build_report.json
      data.yaml
      data_test_stress.yaml
      images/<split>/
      labels/<split>/

Uma versão materializada não pode ser reconstruída no mesmo ID. Crie dataset_canaleta_v2 para alterar revisão ou splits.

## 9. Treinar

Na versão materializada:

1. Escolha modelo base, épocas, imgsz, batch, device e demais parâmetros.
2. Inicie o treinamento.
3. A fila executa um treinamento por vez.

Cada execução recebe um ID próprio:

    runs/detect/t_20260721T203000_a1b2c3/

O diretório contém workflow_job.json, train.log, args.yaml, results.csv, gráficos e weights/. A mesma versão pode ser treinada repetidamente com parâmetros diferentes.

## 10. Excluir recursos

Ao excluir:

- consulte as versões e treinamentos dependentes;
- decida se deseja cascata;
- para origens, decida se os vídeos físicos também devem ser apagados;
- verifique o aviso sobre vídeos compartilhados;
- digite exatamente o ID solicitado.

Cancelar a cascata não cancela necessariamente a exclusão do recurso principal: dependentes podem ser preservados e ficar inválidos por decisão do usuário.

## Checklist final

- import_tasks.json foi gerado uma única vez;
- a exportação correta virou revisão;
- todos os vídeos estão em um único split;
- manifest.csv e build_report.json existem;
- workflow_job.json aponta para a versão correta;
- best.pt pertence ao training_id esperado.
