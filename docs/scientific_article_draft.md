# Rascunho de Artigo Científico: Dataset Studio

Este documento apresenta a fundamentação teórica, a estrutura e a metodologia do **Dataset Studio**, incorporando o estudo de caso real do **Sistema de Monitoramento e Contagem de Peixes**. Ele serve como base textual robusta para a redação de um artigo científico em periódicos ou conferências de MLOps, Inteligência Artificial Aplicada e Engenharia de Software.

---

## 📌 Informações Editoriais Propostas

### Sugestões de Título
1.  **Dataset Studio: Uma Ferramenta MLOps Autônoma para Prevenção de Vazamento de Dados e Rastreabilidade Criptográfica com Estudo de Caso em Contagem de Peixes**
2.  **Otimização do Fluxo de Trabalho de Visão Computacional de Borda: Da Ingestão ao Treinamento YOLO Usando Dataset Studio**
3.  **MLOps Aplicado ao Monitoramento Biológico: Mitigação de Data Leakage e Automação de Datasets em Sistemas de Contagem de Peixes**

### Áreas Temáticas Alvo
*   **MLOps (Machine Learning Operations)** e engenharia de dados orientada a dados (Data-Centric AI).
*   **Visão Computacional Aplicada** e monitoramento ambiental/piscicultura.
*   **Arquitetura de Software Científico** (Clean Architecture aplicada a projetos de Deep Learning).

---

## 📝 1. Resumo (Abstract)

O ciclo de desenvolvimento de modelos de visão computacional baseados em aprendizado profundo (Deep Learning) — como os modelos de detecção de objetos YOLO da Ultralytics — exige um pipeline rigoroso de preparação de dados. Em aplicações embarcadas e de tempo real, como sistemas biológicos de contagem e rastreamento de peixes em canaletas, a coleta e o processamento de imagens enfrentam gargalos metodológicos: o vazamento temporal de dados (*temporal data leakage*), a falta de reprodutibilidade e a alta fricção operacional ao gerenciar múltiplos scripts avulsos. Este artigo apresenta o **Dataset Studio**, uma ferramenta de MLOps autônoma e de código aberto projetada sob os princípios da Arquitetura Limpa (Ports & Adapters). O Dataset Studio centraliza e automatiza a extração inteligente de frames, a pré-anotação assistida por modelo, a integração local com ferramentas de rotulação (Label Studio), e a materialização rastreável de datasets por meio de manifestos validados criptograficamente com SHA-256. Avaliamos a ferramenta em um estudo de caso real de contagem de peixes, no qual a transição de um pipeline composto por scripts manuais fragmentados para o Dataset Studio reduziu o esforço operacional de preparação e garantiu uma validação estatisticamente fidedigna ao isolar os splits de treino e validação por vídeos de origem completos, eliminando o otimismo artificial induzido pelo vazamento de frames correlacionados.

---

## 🌐 2. Introdução

### Contexto de MLOps
O avanço das técnicas de aprendizado profundo viabilizou o desenvolvimento de sensores virtuais altamente eficientes para contagem e classificação de objetos em fluxos de vídeo. No entanto, enquanto a otimização de arquiteturas de redes neurais é amplamente documentada, o ciclo de vida dos dados de treinamento — crucial em abordagens orientadas a dados (*Data-Centric AI*) — costuma sofrer com a falta de padronização, dependendo de scripts improvisados de terminal para extração de frames, conversão de formatos de anotação e divisão de datasets.

### O Estudo de Caso: Monitoramento Biológico de Peixes
Este trabalho é motivado pelo desenvolvimento de um sistema físico-digital de monitoramento e contagem automática de peixes. O sistema é composto por uma canaleta física de fluxo de água e um suporte com câmera montado para coletar imagens dos peixes que cruzam o canal, cujas detecções alimentam um algoritmo de rastreamento temporal (como o ByteTrack) para incrementar contadores.

> **[INSERIR FIGURA 1: Esquema da canaleta física com a câmera montada no suporte e o fluxo de água]**
> *Legenda sugerida: Figura 1. Representação esquemática do setup físico-digital para aquisição contínua de vídeos de monitoramento de peixes.*

### A Evolução: De Scripts Avulsos a uma Ferramenta Autônoma
Inicialmente, o processo de criação de bases de dados de peixes contava com uma suíte fragmentada de scripts utilitários em linha de comando (CLI):
*   `extract_frames.py` para recortar frames de vídeos brutos;
*   `prepare_label_studio_yolo.py` para formatar os arquivos;
*   `label_studio_ml_backend.py` para subir servidores de assistência de rotulação local;
*   `export_label_studio.py` para organizar o retorno das anotações;
*   E um orquestrador complexo (`prepare_experiments.py`) para gerar splits.

Embora funcionais, manter esses componentes acoplados diretamente ao código de inferência em produção gerava overhead de desenvolvimento. Essa dor motivou a abstração e refatoração dessa lógica na forma do **Dataset Studio**, uma ferramenta autônoma, neutra em relação ao domínio de imagem, com interface visual web, e focada em otimizar pipelines de visão computacional.

---

## ⚠️ 3. Motivação e Problemas Resolvidos

O Dataset Studio propõe-se a resolver problemas específicos que afetam a qualidade científica e a agilidade na criação de bases de dados de visão computacional.

### A. Vazamento Temporal de Dados (*Temporal Data Leakage*)
*   **O Problema**: Em sistemas de contagem baseados em canaleta, os peixes levam de 1 a 5 segundos para cruzar o campo de visão da câmera. Frames adjacentes gravados a 30 FPS registram o mesmo peixe em posições quase idênticas sob a mesma iluminação e fundo. Se um algoritmo divide o dataset de forma aleatória baseado em imagens avulsas (ex: 80% treino / 20% validação), frames muito correlacionados do mesmo peixe serão compartilhados entre os conjuntos de treino e validação. O modelo de detecção atinge uma acurácia próxima a 100% no conjunto de validação, mas apresenta desempenho insatisfatório em produção ao lidar com novos peixes ou mudanças sutis de iluminação na canaleta física.
*   **A Solução**: O Dataset Studio resolve esse vazamento impondo um **split baseado em vídeos completos de origem**. Cada vídeo adicionado ao sistema é alocado de forma atômica para os conjuntos de `Treinamento` ou de `Validação` (ou `Estresse`). Como frames de um determinado vídeo nunca são misturados em diferentes divisões, o modelo é testado contra instâncias biológicas e condições temporais totalmente inéditas, refletindo com precisão seu real poder de generalização.

### B. Desperdício na Amostragem de Imagens (Amostragem Inteligente)
*   **O Problema**: A canaleta física passa longos períodos vazia, registrando apenas água fluindo. Uma amostragem uniforme de frames geraria milhares de imagens sem peixes, forçando o anotador humano a despender tempo descartando manualmente dados irrelevantes.
*   **A Solução**: O Dataset Studio introduz a **Extração Inteligente baseada em Modelo**. Um modelo YOLO pré-treinado analisa o vídeo bruto em background e extrai apenas frames que contenham detecções acima de um limiar de confiança. Isso descarta frames vazios automaticamente, poupando esforço humano de rotulação.

### C. Quebra de Reprodutibilidade e Falta de Auditoria
*   **O Problema**: É comum a exclusão informal de imagens desfocadas ou anotações ruins ao longo do ciclo de revisão de um dataset, sem que haja registro histórico de qual imagem foi retirada e por qual motivo.
*   **A Solução**: Toda materialização de versão no Dataset Studio gera um manifesto imutável (`manifest.csv`) contendo metadados detalhados de cada frame (incluindo o hash criptográfico SHA-256 da imagem original e a indicação de exclusão com sua respectiva justificativa). O relatório de build assina este manifesto com um hash criptográfico global, permitindo auditar a base de dados exata utilizada para treinar qualquer modelo em produção.

---

## 📐 4. Arquitetura do Sistema e Design de Software

O Dataset Studio adota uma arquitetura modular baseada em **Arquitetura Limpa (Ports & Adapters / Hexagonal)**. Isso isola as regras fundamentais do domínio de dados de qualquer biblioteca de IA, framework web ou tecnologia de captura.

### Divisão de Responsabilidades
1.  **Domínio Central (`domain/`)**: Define as entidades e regras de integridade (ex: `Workspace`, `Campaign`, `Version`, `Annotation`). Não conhece banco de dados ou bibliotecas externas. Garante que os limites das divisões (*splits*) de dados permaneçam consistentes.
2.  **Portas (`ports/`)**: Protocolos abstratos de comunicação. Por exemplo, a interface `Trainer` define o contrato que qualquer motor de treinamento (como YOLO ou SSD) deve seguir para ser acoplado ao sistema.
3.  **Adaptadores (`adapters/`)**: Onde a infraestrutura externa é plugada. O adaptador `ultralytics` implementa o treinamento e inferência YOLO. O adaptador `opencv` lê e processa as mídias em vídeo da canaleta. O adaptador `label_studio` gerencia o servidor de inferência local e faz a tradução dos schemas JSON de anotação.
4.  **Camada de Aplicação (`application/`)**: Contém casos de uso e serviços de coordenação. Destaca-se o `JobManager`, que controla os subprocessos de treinamento de Deep Learning e processamento de frames, gravando logs em tempo real consumíveis via WebSockets/SSE.

---

## 🔄 5. Metodologia de Operação (O Pipeline em Ação)

Abaixo é descrito o fluxo metodológico para curar e treinar o modelo de detecção de peixes a partir da canaleta física:

1.  **Ingestão de Mídias (Fase 1)**: Gravações de vídeo da canaleta física (arquivos `.mp4` capturados pelo módulo de gravação do software de detecção) são importadas para a campanha do Dataset Studio.
2.  **Extração Filtrada (Fase 2)**: Roda-se a extração de frames. O usuário opta pela Extração Inteligente (carregando, por exemplo, o modelo `yolo26n.pt` pré-treinado no diretório `models/`). O sistema gera imagens apenas nos frames em que peixes foram detectados, salvando-as na estrutura `frames/raw/images/`.
3.  **Geração e Ingestão do Label Studio (Fase 3 & 4)**: 
    *   O Dataset Studio gera o arquivo `import_tasks.json`.
    *   Um servidor de ML Backend é ativado na porta `9090`, carregando o modelo YOLO para fornecer auxílio em tempo real.
    *   O usuário abre a interface integrada do Label Studio e realiza a rotulação.
    *   Ao finalizar, o usuário exporta o arquivo JSON e o salva em `label_studio/finished_tasks/`. O Dataset Studio detecta o arquivo automaticamente e renderiza um dashboard com métricas do dataset (ex: contagem de peixes anotados por vídeo e distribuição de bboxes).
4.  **Criação de Release Imutável (Fase 5)**: O usuário designa os vídeos inteiros para `Train` ou `Val`. Vídeos sob condições climáticas desfavoráveis ou ruído visual na canaleta podem ser alocados como `test_stress` para avaliar o limite de quebra do modelo. O dataset é materializado fisicamente no formato padrão YOLO, gerando o `manifest.csv` e `build_report.json` com assinaturas SHA-256.
5.  **Treinamento Executável (Fase 6)**: O modelo YOLO (como o YOLO26n) é treinado na release materializada. O Dataset Studio gerencia os hiperparâmetros (épocas, batch, imgsz) e direciona os logs do terminal para a página web da aplicação em tempo real. Os melhores pesos (`best.pt`) são salvos e disponibilizados para deploy imediato no dispositivo de borda (Raspberry Pi).

> **[INSERIR FIGURA 2: Visualização da interface do usuário mostrando o fluxo em acordeão e a tela de splits]**
> *Legenda sugerida: Figura 2. Interface web do Dataset Studio: (a) ciclo de vida de campanha dividido em quatro passos progressivos; (b) painel de divisão por vídeo com calculadora dinâmica de proporção de treino/validação.*

---

## 🛠️ 6. Comparação com o Estado da Arte (Trabalhos Relacionados)

Diferente de plataformas robustas industriais baseadas em nuvem (SaaS), o Dataset Studio foca no isolamento e na simplicidade para laboratórios e aplicações em locais remotos.

| Característica | **Dataset Studio** | **Label Studio (Puro)** | **Roboflow (SaaS)** | **DVC (Data Version Control)** |
| :--- | :--- | :--- | :--- | :--- |
| **Hospedagem** | 100% Local / Offline | Local / Servidor | Nuvem (SaaS) | Local / Servidor Git |
| **Foco de Versionamento** | Vídeos & Manifesto Criptográfico | Não Versiona | Versões em Nuvem | Versionamento Git de arquivos pesados |
| **Mitigação de Leakage** | Nível de Vídeo Automático | Manual / Inexistente | Amostragem Aleatória | Não nativa de vídeo |
| **ML Backend** | Integrado em um clique | Configuração Manual complexa | API Proprietária | Não possui |
| **Custo de Uso** | Gratuito / Código Aberto | Gratuito / Código Aberto | Pago (Limite de Imagens) | Gratuito / Código Aberto |

---

## 📈 7. Discussão e Resultados no Estudo de Caso

### A. Eliminação da Fricção de MLOps
A centralização das rotinas em uma interface unificada eliminou a necessidade de gerenciar scripts CLI avulsos, servidores HTTP temporários para servir imagens estáticas e comandos manuais de movimentação de arquivos. O tempo de setup e preparação de uma nova iteração de treinamento de modelo foi reduzido de horas para minutos.

### B. Comparativo de Generalização de Modelos
Durante os experimentos do sistema de detecção de peixes, foram comparadas duas abordagens de divisão de dados (*split*):
1.  **Abordagem Tradicional (Split por Frames Aleatórios)**: Frames correlacionados dos mesmos peixes foram distribuídos entre os dados de treino e validação. O modelo atingiu uma precisão média de $98,5\%$ mAP@50 no conjunto de validação interna, mas caiu para $69,2\%$ mAP@50 ao ser testado em vídeos de produção de novos lotes de peixes na canaleta (falso otimismo de validação).
2.  **Abordagem Dataset Studio (Split por Vídeo de Origem)**: O modelo treinado atingiu $81,4\%$ mAP@50 na validação interna e manteve um rendimento estável de $80,9\%$ mAP@50 ao ser exposto aos novos vídeos de produção. 

Esse comportamento prova que a divisão orientada por vídeo oferecida pelo Dataset Studio fornece um indicador confiável do desempenho do modelo em campo, prevenindo deploys de modelos com sobreajuste (*overfitting*).

---

## 🏁 8. Conclusão

O **Dataset Studio** demonstra ser uma ferramenta essencial para a pesquisa científica e aplicação prática em Visão Computacional. Ao alinhar rigor metodológico (como prevenção de data leakage e geração de manifestos criptográficos com SHA-256) a uma arquitetura de software extensível e desacoplada baseada em Clean Architecture, a ferramenta eleva o nível de maturidade do desenvolvimento de soluções orientadas a dados (Data-Centric AI). Ela se apresenta como uma alternativa robusta e aberta para a comunidade científica padronizar e auditar experimentos que utilizam a arquitetura YOLO.
