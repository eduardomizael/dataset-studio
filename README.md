# Dataset Studio 🚀

Ferramenta autônoma para organização, revisão, materialização de datasets de visão computacional e treinamento configurável de modelos YOLO.

---

## ⚡ Como Executar

No terminal, acesse o diretório do repositório `dataset-studio` e execute:

```bash
uv run dataset-studio.py
```

A aplicação iniciará o servidor local e abrirá automaticamente o navegador na interface visual: `http://127.0.0.1:8000/`.

---

## 📚 Documentação Oficial

Toda a documentação do projeto está disponível na pasta [`docs/`](docs/):

1. 📖 **[Manual do Usuário](docs/USER_MANUAL.md)**: Guia passo-a-passo sobre como criar Origens de dados (`sources`), extrair frames (modo uniforme ou inteligente), integrar com o Label Studio, gerar Versões (`versions`) com splits sem vazamento de mídias e treinar modelos YOLO.
2. 📐 **[Estrutura e Arquitetura](docs/ARCHITECTURE_AND_STRUCTURE.md)**: Mapeamento completo dos diretórios do repositório, Clean Architecture (Domínio, Aplicação, Adaptadores) e fluxo de dados no workspace (`dataset/sources/`, `dataset/versions/`, `models/`, `runs/`).
3. 📝 **[Checklist de Documentação (TODO)](docs/DOCUMENTATION_TODO.md)**: Backlog de tarefas de documentação do projeto.

---

## 🧪 Suíte de Testes

Para rodar todos os testes automatizados unitários, de caracterização e de integração:

```bash
uv run pytest
```

---

## 📄 Licença

Este projeto está licenciado sob a licença [MIT](LICENSE).
