# Checklist (TODO) para Finalização do Artigo Científico

Este documento detalha as etapas práticas que você deve seguir para concluir e submeter o artigo científico sobre o **Dataset Studio** com base no estudo de caso de contagem de peixes.

---

## 📸 1. Captura de Figuras e Telas (Visual System Design)
Artigos de ferramentas de software em engenharia de software e visão computacional (*Tool Demo Papers* ou *System Track*) exigem evidências visuais de funcionamento.

- [ ] **Figura 1: A Canaleta Física e Setup de Coleta**
  - Capturar uma foto ou desenho esquemático do aparato físico (canaleta de fluxo, câmera suspensa, iluminação e o computador de borda/Raspberry Pi). Isso contextualiza o leitor no domínio do problema.
- [ ] **Figura 2: Dashboard Inicial do Dataset Studio**
  - Captura de tela do painel inicial mostrando as três colunas: Campanhas (Sources), Versões Materializadas (Releases) e os Treinamentos ativos.
- [ ] **Figura 3: O Fluxo das 4 Etapas (Acordeão)**
  - Capturar a tela do ciclo de vida de uma origem, exibindo extração, fixação, integração com Label Studio e revisões aceitas explicitamente.
- [ ] **Figura 4: Tela de Configuração do Split por Vídeo**
  - Capturar a montagem da versão, mostrando a calculadora enquanto os vídeos são atribuídos a `train`, `val`, `test_normal` e `test_stress`.
- [ ] **Figura 5: Terminal de Monitoramento de Treino**
  - Print da interface que exibe as saídas de logs do YOLO executado em segundo plano de forma interativa.

---

## 📊 2. Experimentos e Benchmarks de MLOps
Para dar rigor científico à seção de "Discussão e Resultados", é crucial colher métricas exatas do seu ambiente físico.

- [ ] **Coletar tempo economizado no Pipeline**:
    - Medir quanto tempo (em minutos/horas) leva o fluxo manual clássico (executar scripts individuais, extrair frames, levantar servidor HTTP estático, exportar zip, descompactar na pasta certa, rodar split CLI) contra o fluxo unificado no Dataset Studio.
- [ ] **Mensurar o efeito da Extração Inteligente**:
    - Gravar 1 hora de vídeo na canaleta (com peixes cruzando de forma esparsa).
    - Anotar o percentual de frames contendo peixes versus frames vazios.
    - Registrar quantos frames o Dataset Studio extraiu no Modo Inteligente versus Modo Uniforme e a taxa de detecções redundantes filtradas.
- [ ] **Executar os Treinamentos das Duas Abordagens**:
    - Treinar o YOLO (ex: `YOLO26n` ou `YOLO11n`) com split puramente aleatório (com vazamento temporal).
    - Treinar o mesmo modelo com split isolado por vídeo (via Dataset Studio).
    - Avaliar ambos os modelos resultantes em uma pasta fixa de validação com novos peixes nunca antes vistos nas gravações de treino. Anotar o mAP@50 de ambos e documentar o hiato de precisão (o falso otimismo do split clássico).

---

## ✍️ 3. Redação e Fundamentação Científica
- [ ] **Seção de Trabalhos Relacionados (Related Works)**:
  - Fazer uma revisão bibliográfica rápida e comparar o Dataset Studio com:
    - *Roboflow / V7 / CVAT (SaaS)*: O Dataset Studio é 100% local, focado em privacidade, livre de custos de API e funciona perfeitamente offline em laboratórios isolados.
    - *DVC (Data Version Control)*: O DVC gerencia blobs no Git, enquanto o Dataset Studio gerencia o ciclo completo visual de anotação, revisão e split de vídeo.
    - *Label Studio Puro*: O Dataset Studio serve os dados locais, gera tarefas, ativa o ML Backend e inspeciona exportações; a criação do projeto e a importação das tasks ainda são ações realizadas no Label Studio.
- [ ] **Equações e Formalismo Matemático**:
  - Definir matematicamente a correlação de frames consecutivos e por que a amostragem independente e identicamente distribuída ($i.i.d.$) é violada quando há correlação temporal entre quadros de um mesmo vídeo.
- [ ] **Revisão de Terminologia**:
  - Garantir o alinhamento de termos. Ex: no banco de dados interno e no código, a ferramenta usa os termos `Sources` e `Versions`. Na interface do usuário, exibe `Campanhas` e `Releases`. O texto do artigo deve deixar claro que `Release` é a materialização física de uma `Version` estruturada.

---

## 📄 4. Formatação e Submissão
- [ ] **Escolha do Evento/Periódico**:
  - *IEEE Latin America Transactions*, *SIBRAPI (Trilha de WUW - Workshop de Visão Computacional)*, *SBC (Simpósio Brasileiro de Computação)* ou simpósios de engenharia de software (como *SBES* na trilha de ferramentas).
- [ ] **Configuração do Template LaTeX (Overleaf)**:
  - Importar o template específico (IEEEtran ou SBC) para o Overleaf.
- [ ] **Tradução para Inglês**:
  - Caso o evento seja de âmbito internacional, traduzir o texto final mantendo termos técnicos como *data leakage*, *edge computing* e *human-in-the-loop*.
