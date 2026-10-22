# Plano v4

Aprovado em 22/10/2026. Objetivo: aproximar o projeto das soluções de topo do
Kaggle sem perder a leveza para o celular. O modelo final é **uma rede do
tamanho da B0 + TTA leve**, e o exame tem as **duas fotos** do paciente.
Fontes: [REFERENCIAS.md](REFERENCIAS.md).

## Ponto de partida

A campanha de outubro (tag `campanha-codex-2026-10-20`, B0 448 com dados
extras) mostrou que ajustes finos empatam dentro do ruído. Só mudaram o
resultado:

- **Normalização de Ben:** sem ela, a especificidade caiu de 0,671 para 0,609.
- **Resolução:** em 384 px, caiu para 0,538.
- **TTA:** com 4 variantes, a especificidade sobe para 0,711, sem treinar nada.

**Referência (E01), validação por paciente com sensibilidade 0,90:** ROC-AUC
0,908 e especificidade 0,671. Com TTA4: 0,913 e 0,711.

## Como executar

1. **Caches (uma vez por tamanho).** Rode `notebooks/Preparar_Caches.ipynb`
   montando o Drive da conta pessoal, uma cópia para `R01_b0_512` e outra para
   `R02_b0_640`. Depois, compartilhe `ProjetoRetinopatia_caches` com a conta do
   projeto como Leitor e crie o atalho em "Meu Drive" da conta do projeto.
2. **Fumaça.** Antes de cada rodada, para cada recurso novo (destilação, pares,
   V2-B0), rode o notebook de treino com `RETINOPATIA_FUMACA = "1"`. Leva
   minutos e confere que treina, salva e recarrega.
3. **Rodada.** Faça até 3 cópias de `notebooks/Retinopatia_0_1.ipynb`, uma por
   experimento (muda só `EXPERIMENTO` na primeira célula), cada uma numa sessão
   **A100 com RAM alta**. Uma execução por sessão.
4. **Acompanhe pelo Drive** (`historico_fase*.csv`, `resumo.csv` em
   `resultados/<experimento>/`), não pela tela do notebook.
5. **Fechamento.** Rode a última célula (`consolidar_experimentos`) numa sessão
   só e o `notebooks/Analisar_Resultados.ipynb` (só CPU). Registre os números na
   tabela abaixo e escolha a rodada seguinte.

O teste fica fechado (`RETINOPATIA_AVALIAR_TESTE = "0"`) até o modelo final.

## Experimentos

Métricas de validação por paciente (especificidade com sensibilidade 0,90). As
rodadas 2 e 3 são definidas depois de ver a anterior.

| ID | Rodada | O que testa | Espec. sem TTA | Espec. TTA2 | Espec. TTA4 | ROC sem TTA | Kappa grau |
|---|---|---|---:|---:|---:|---:|---:|
| E01 | ref. | B0 448 (campanha de outubro) | 0,671 | — | 0,711 | 0,908 | 0,804 |
| R01 | 1 | B0 512 | 0,686 | 0,705 | 0,720 | 0,914 | 0,820 |
| **R02** | 1 | **B0 640** | **0,732** | 0,723 | 0,714 | **0,923** | 0,818 |
| R03 | 1 | Destilação B4 448 → B0 448 | 0,609 | 0,605 | 0,686 | 0,904 | 0,809 |
| R04 | 2 | B0 com pares de olhos, 640 px (só se o combinador mostrar ganho) | | | | | |
| R05 | 2 | EfficientNetV2-B0, 640 px | | | | | |
| R06 | 2 | B0 640 com treino mais longo (18 épocas na Fase 2) | | | | | |
| R07 | 3 | Combinação das alavancas vencedoras | | | | | |
| R08–R09 | 3 | R07 com sementes 17 e 73 (mesmo split) | | | | | |
| R10 | reserva | 768 px ou decisão ordinal | | | | | |

### Leva 1 (22/10/2026, commit `3e3dfd9`, 3 sessões A100 em paralelo)

Validação, por paciente. ROC-AUC sem TTA; kappa da cabeça de grau por olho.

- **R02 (640 px) venceu com folga:** ROC-AUC 0,923 (+0,016 sobre o E01) e
  especificidade 0,732 sem TTA (+6 pontos). Por olho, ROC-AUC 0,927 (0,931 com
  TTA4). É o nível da B4 448 com dados extras, numa rede do tamanho da B0. A
  melhor época foi a última (12 de 12): ainda estava melhorando.
- **R01 (512 px):** ROC-AUC 0,914 e especificidade 0,686 sem TTA (0,720 com
  TTA4). Ganho menor, na mesma direção.
- **R03 (destilação a 448 px) piorou:** especificidade 0,609 sem TTA. O professor
  foi treinado nas mesmas imagens e dá probabilidades confiantes demais no treino;
  a destilação sai das próximas levas.
- **De onde veio o ganho:** com 90% de sensibilidade, a fração de olhos de grau 0
  marcados como doentes caiu de 34,8% (E01) para 25,5% (R02). A sensibilidade no
  grau 1 não mudou (72% no E01, 69% no R02); nos graus 2 a 4 ficou em 96–100%.
- **TTA a 640 px** quase não muda a ROC-AUC (0,923 → 0,926) e a especificidade
  no ponto de 90% oscila (0,732 sem TTA, 0,714 com TTA4). No celular, a 640 px,
  TTA deixa de ser obrigatório.
- **Custo:** a 640 px, cada época levou ~11 min (A100), mais ~1 h para processar
  as imagens, porque ainda não há cache de 640. O R02 levou cerca de 4 h.

Diferenças de cerca de 0,02 a 0,03 na especificidade ficam dentro do ruído de
uma semente só (cerca de 1.200 pacientes sem a doença na validação). Os
finalistas são repetidos com outras sementes antes da decisão.

## Final

1. Avaliar o modelo escolhido uma única vez no teste (`RETINOPATIA_AVALIAR_TESTE = "1"`).
2. Escolher TTA2 ou TTA4 pela validação.
3. Exportar para TFLite (fp16 e int8) e conferir tamanho e paridade com o Keras.
4. Adaptar a API e o app para duas fotos por exame (tarefa separada).
