# Checklist e Backlog de Documentação - Dataset Studio 📝

Este documento rastreia todas as tarefas de documentação do repositório `dataset-studio`, divididas por prioridade e componentes.

---

## 1. Documentação Principal (Concluída)
- [x] **Manual do Usuário (`docs/USER_MANUAL.md`)**: Guia passo-a-passo cobrindo inicialização, campanhas, extração uniforme/inteligente, integração com Label Studio via `finished_tasks/`, calculadora de splits em tempo real e treinamento YOLO.
- [x] **Estrutura e Arquitetura (`docs/ARCHITECTURE_AND_STRUCTURE.md`)**: Diagrama de camadas Hexagonal/Clean Architecture, mapa de pastas `src/`, diretórios de dados dinâmicos do workspace (`campaigns/`, `releases/`, `models/`, `runs/`).
- [x] **README Principal (`README.md`)**: Visão geral do repositório, comandos rápidos de execução (`uv run dataset-studio.py`) e links para a documentação técnica.

---

## 2. Documentação de API e Referência Técnica (Concluída)
- [x] **Documentação da API REST (`docs/API_REFERENCE.md`)**: Especificação detalhada dos endpoints FastAPI (`/api/campaigns`, `/api/releases`, `/api/jobs`, `/api/trainings`).
- [x] **Guia do CLI (`docs/CLI_REFERENCE.md`)**: Especificação dos subcomandos da linha de comando (`dataset-studio campaign`, `dataset-studio release train`).
- [x] **Guia de Adaptadores Customizados (`docs/ADAPTERS_GUIDE.md`)**: Como implementar novas portas de predição (`Predictor`) ou novos frameworks de treino além do Ultralytics YOLO.

---

## 3. Tutoriais e Exemplos (Concluída)
- [x] **Exemplo End-to-End (`docs/TUTORIAL_E2E.md`)**: Walkthrough passo a passo com amostras de dados fictícios para treinamento completo de um modelo de peixes a partir do zero.
- [x] **Guia de Resolução de Problemas (`docs/TROUBLESHOOTING.md`)**: Resolução de dúvidas comuns sobre portas ocupadas (8000/9090), instalação do `uv` e dependências OpenCV/PyTorch.
