# Checklist e Backlog de Documentação - Dataset Studio 📝

Este documento rastreia todas as tarefas de documentação do repositório `dataset-studio`, divididas por prioridade e componentes.

---

## 1. Documentação Principal (Concluída)
- [x] **Manual do Usuário (`docs/USER_MANUAL.md`)**: Ciclo source/revision/version/training, quatro splits, imutabilidade e exclusão consciente.
- [x] **Estrutura e Arquitetura (`docs/ARCHITECTURE_AND_STRUCTURE.md`)**: Camadas, `dataset/sources/`, `dataset/versions/`, staging, hashes, jobs e autonomia.
- [x] **README Principal (`README.md`)**: Instalação com `uv sync --all-extras`, ciclo de vida e índice da documentação.

---

## 2. Documentação de API e Referência Técnica (Concluída)
- [x] **Documentação da API REST (`docs/API_REFERENCE.md`)**: Rotas reais de sources, versions, revisions, trainings, jobs e deletion-impact.
- [x] **Guia do CLI (`docs/CLI_REFERENCE.md`)**: Comandos canônicos `source` e `version`, aliases e limitações atuais.
- [x] **Guia de Adaptadores Customizados (`docs/ADAPTERS_GUIDE.md`)**: Como implementar novas portas de predição (`Predictor`) ou novos frameworks de treino além do Ultralytics YOLO.

---

## 3. Tutoriais e Exemplos (Concluída)
- [x] **Exemplo End-to-End (`docs/TUTORIAL_E2E.md`)**: Walkthrough passo a passo com amostras de dados fictícios para treinamento completo de um modelo de peixes a partir do zero.
- [x] **Guia de Resolução de Problemas (`docs/TROUBLESHOOTING.md`)**: Portas 8000/8080/9090, CUDA 12.8, codecs, staging e recursos órfãos.

---

## 4. Backlog

- [ ] Criar `CONTRIBUTING.md` com ambiente, Ruff, Mypy, testes, branches e Pull Requests.
- [ ] Documentar criação automática de projeto, storage e registro do ML Backend quando essa automação for implementada.
- [ ] Adicionar exemplos versionados de payloads da API em `docs/examples/`.
