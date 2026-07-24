# Dataset Studio 🚀

Ferramenta autônoma para organização, revisão, materialização de datasets de visão computacional e treinamento configurável de modelos YOLO.

---

## ⚡ Como Executar

No terminal, acesse o diretório do repositório `dataset-studio` e execute:

```bash
uv sync --all-extras
uv run --all-extras dataset-studio.py
```

O primeiro comando instala também o Label Studio, Ultralytics e as dependências
de desenvolvimento no ambiente isolado deste repositório. A ferramenta não usa
o Python nem o YOLO de outros projetos. No Windows, o extra `cuda` instala os
wheels oficiais CUDA 12.8 do PyTorch; em máquinas sem GPU, selecione `device=cpu`.

A aplicação iniciará o servidor local e abrirá automaticamente o navegador na interface visual: `http://127.0.0.1:8000/`.

## Ciclo de vida

1. Uma **origem** recebe vídeos, pode dividi-los em unidades experimentais
   virtuais e registra a configuração de extração.
2. A geração de `import_tasks.json` fixa a origem.
3. Exportações do Label Studio geram revisões independentes.
4. Uma **versão** seleciona revisões e atribui unidades experimentais inteiras
   a `train`, `val`, `test_normal` e `test_stress`.
5. A materialização publica o dataset em `dataset/versions/<version_id>/`.
6. A mesma versão materializada pode alimentar múltiplos treinamentos em `runs/detect/<training_id>/`.
7. Cada treinamento avalia automaticamente o `best.pt` em `test_normal` e
   `test_stress` e apresenta a variação das métricas (`estresse - normal`).

A integração com o Label Studio exige um token somente na primeira utilização do computador. Depois disso, o Dataset Studio cria ou reconhece o projeto de cada origem, importa tarefas sem duplicação e configura automaticamente a fila e as preanotações.

Recursos podem ser excluídos conscientemente. A interface mostra dependências, oferece cascata, permite preservar vídeos e exige confirmação digitada; o usuário pode prosseguir mesmo quando a decisão deixar dependentes inválidos.

---

## 📚 Documentação Oficial

Toda a documentação do projeto está disponível na pasta [`docs/`](docs/):

1. 📖 **[Manual do Usuário](docs/USER_MANUAL.md)**: Guia passo-a-passo sobre como criar Origens de dados (`sources`), extrair frames (modo uniforme ou inteligente), integrar com o Label Studio, gerar Versões (`versions`) com splits sem vazamento de mídias e treinar modelos YOLO.
2. 📐 **[Estrutura e Arquitetura](docs/ARCHITECTURE_AND_STRUCTURE.md)**: Mapeamento completo dos diretórios do repositório, Clean Architecture (Domínio, Aplicação, Adaptadores) e fluxo de dados no workspace (`dataset/sources/`, `dataset/versions/`, `models/`, `runs/`).
3. 🧭 **[Tutorial End-to-End](docs/TUTORIAL_E2E.md)**: Ciclo completo com Label Studio, quatro splits e treinamento.
4. 🌐 **[Referência da API](docs/API_REFERENCE.md)**: Rotas, payloads, exclusões e jobs.
5. ⌨️ **[Referência da CLI](docs/CLI_REFERENCE.md)**: Comandos disponíveis e limitações atuais.
6. 🛠️ **[Troubleshooting](docs/TROUBLESHOOTING.md)**: Portas, CUDA, codecs, staging e recursos órfãos.
7. 🔌 **[Guia de Adaptadores](docs/ADAPTERS_GUIDE.md)**: Contratos de predição e treinamento.
8. 🧬 **[Catálogo de Linhagem](docs/MODEL_REGISTRY.md)**: Origens, datasets,
   treinamentos, modelos, aliases, manifests retroativos e validação por
   SHA-256.
9. 🗄️ **[Arquivo de Datasets Legados](docs/LEGACY_ARCHIVE.md)**: Preservação física deduplicada, verificação e reconstrução de snapshots históricos.

---

## 🧪 Suíte de Testes

Para rodar todos os testes automatizados unitários, de caracterização e de integração:

```bash
uv run --all-extras pytest
```

---

## 📄 Licença

Este projeto está licenciado sob a licença [MIT](LICENSE).
