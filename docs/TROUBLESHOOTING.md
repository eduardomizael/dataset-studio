# Resolução de problemas

## Instalação recomendada

Sempre comece pela raiz do repositório:

~~~powershell
uv sync --all-extras
uv run --all-extras dataset-studio.py
~~~

O uso de --all-extras é necessário para instalar Label Studio, Ultralytics, PyTorch CUDA no Windows e dependências de teste.

## Portas ocupadas

Portas padrão:

- 8000: Dataset Studio;
- 8080: Label Studio;
- 9090: ML Backend.

No PowerShell:

~~~powershell
Get-NetTCPConnection -LocalPort 8000,8080,9090 -ErrorAction SilentlyContinue |
  Select-Object LocalAddress,LocalPort,State,OwningProcess
~~~

Verifique o processo antes de encerrá-lo:

~~~powershell
Get-Process -Id <PID> | Select-Object Id,ProcessName,Path
Stop-Process -Id <PID>
~~~

Para alterar apenas a porta do painel:

~~~powershell
uv run --all-extras dataset-studio.py --port 8010
~~~

As portas 8080 e 9090 ainda são fixas na integração atual.

## Label Studio não inicia

Confirme:

~~~powershell
uv run --all-extras label-studio --version
~~~

Se o executável estiver ausente, repita uv sync --all-extras. O Dataset Studio procura primeiro no mesmo ambiente Python e depois no PATH.

O Label Studio abre normalmente na tela de login. Uma resposta HTTP 200 em /user/login/ confirma que o servidor está disponível.

## Integração automática pede token

Isso ocorre somente na primeira utilização do Label Studio neste computador:

1. inicie o Label Studio;
2. faça login;
3. abra `Account & Settings > Access Token`;
4. copie um token;
5. cole no painel Integração automática do Dataset Studio;
6. clique em Salvar e preparar esta origem.

O token é usado pela API oficial e não é salvo dentro da origem ou do repositório. Também é possível fornecê-lo por ambiente:

~~~powershell
$env:DATASET_STUDIO_LABEL_STUDIO_API_KEY = "seu-token"
$env:DATASET_STUDIO_LABEL_STUDIO_URL = "http://127.0.0.1:8080"
~~~

## Label All Tasks abre sem caixas

O Dataset Studio verifica automaticamente:

- se existem predições no `import_tasks.json`;
- quais versões de modelo aparecem;
- quantas tarefas cada versão cobre;
- se as preanotações estão habilitadas;
- se `model_version` corresponde à versão selecionada.

A seleção prioriza cobertura completa, mesmo que exista uma versão mais recente em apenas parte das tarefas. Se nenhuma versão cobrir tudo, a preparação é interrompida e a interface informa a cobertura exata.

Não altere o SQLite do Label Studio no fluxo normal. Use o botão de preparação novamente para reaplicar as configurações pela API oficial.

Importante: `Label All Tasks` abre a próxima tarefa pendente da fila; clicar diretamente numa linha abre aquela tarefa específica e pode mostrar uma anotação humana já concluída.

## ML Backend não fica saudável

O Dataset Studio consulta http://127.0.0.1:9090/health antes de declarar sucesso.

Verifique:

~~~powershell
Invoke-RestMethod http://127.0.0.1:9090/health
~~~

Uma resposta válida contém status UP e model_version.

Se falhar:

- confirme que o modelo está em models/;
- confirme que source.yaml possui classes corretas;
- revise annotation.model e annotation.detection_config;
- verifique CUDA e memória;
- confirme que a porta 9090 não pertence a outro serviço.

## CUDA não é reconhecida

Diagnóstico:

~~~powershell
nvidia-smi
uv run --all-extras python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
~~~

No Windows com NVIDIA, a versão esperada do torch deve possuir sufixo +cu128. Se aparecer +cpu:

~~~powershell
uv lock
uv sync --all-extras
~~~

O pyproject.toml direciona torch e torchvision para o índice oficial pytorch-cu128 quando o extra cuda está ativo.

Para falta de VRAM:

- reduza batch;
- reduza imgsz;
- feche outros processos CUDA;
- use device=cpu somente como fallback.

## OpenCV não abre o vídeo

Teste:

~~~python
import cv2

cap = cv2.VideoCapture("video.mp4")
print(cap.isOpened())
ok, frame = cap.read()
print(ok, None if frame is None else frame.shape)
cap.release()
~~~

Se falhar, converta para MP4/H.264 com FFmpeg:

~~~powershell
ffmpeg -i entrada.avi -c:v libx264 -an saida.mp4
~~~

## Upload rejeitado

Uploads são recusados quando:

- o ID é inválido;
- a origem já existe;
- o nome contém diretórios, como ../video.mp4;
- dois arquivos têm o mesmo nome;
- o staging não consegue ser publicado.

Novas origens ficam em videos/<source_id>/. Um arquivo de outra origem não pode ser sobrescrito por coincidência de nome.

## Extração não pode ser repetida

Depois de existirem frames, a interface bloqueia a etapa. Depois de import_tasks.json existir, a origem está fixada também no domínio.

Isso não é falha. Para mudar configuração:

1. revise o impacto da exclusão;
2. exclua a origem;
3. escolha se preserva ou remove os vídeos;
4. crie uma nova origem.

## Exportação não aparece

O caminho correto é:

    dataset/sources/<source_id>/label_studio/finished_tasks/<arquivo>.json

Use JSON nativo do Label Studio, não JSON-MIN. Atualize a página depois de concluir a cópia.

O arquivo aparecer na inspeção, mas só vira revisão quando o usuário o aceita explicitamente.

## Versão não materializa

Verifique:

- revisão válida para cada origem;
- todos os vídeos atribuídos exatamente uma vez;
- train e val com frames utilizáveis;
- imagens do frame_manifest.json presentes;
- espaço livre para staging e cópia final.

Diretórios temporários usam nomes .<version_id>.build-<token>. Em falha controlada, são removidos. Se a máquina desligar durante a troca final, não apague manualmente backup e staging antes de inspecioná-los.

## Treinamento aparece sem versão

Treinamentos novos persistem release_id/version_id em workflow_job.json. Para runs legados, a API tenta inferir a versão pelo campo data do args.yaml.

Se ambos estiverem ausentes, os pesos continuam acessíveis, mas a proveniência não pode ser reconstruída automaticamente.

## Exclusão deixou dependentes inválidos

Isso pode acontecer quando o usuário exclui sem cascata. A ação é permitida deliberadamente.

Opções:

- excluir também os recursos órfãos;
- restaurar o recurso removido a partir de backup;
- recriar uma versão com revisão existente;
- manter o treinamento apenas como artefato histórico, reconhecendo a perda de proveniência.

Antes de excluir, consulte /api/deletion-impact/<tipo>/<id>.

## Validação do repositório

~~~powershell
uv run --all-extras pytest -q
uv run --all-extras ruff check src tests
git diff --check
~~~
