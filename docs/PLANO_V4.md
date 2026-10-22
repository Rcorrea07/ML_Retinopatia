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
| E01 | ref. | B0 448 (campanha de outubro) | 0,671 | — | 0,711 | 0,908 | — |
| R01 | 1 | B0 512 | | | | | |
| R02 | 1 | B0 640 | | | | | |
| R03 | 1 | Destilação B4 448 → B0 448 | | | | | |
| R04 | 2 | B0 com pares de olhos, na melhor resolução | | | | | |
| R05 | 2 | EfficientNetV2-B0, na melhor resolução | | | | | |
| R06 | 2 | Destilação na melhor resolução, ou progressivo/treino longo | | | | | |
| R07 | 3 | Combinação das alavancas vencedoras | | | | | |
| R08–R09 | 3 | R07 com sementes 17 e 73 (mesmo split) | | | | | |
| R10 | reserva | 768 px ou decisão ordinal | | | | | |

Diferenças de cerca de 0,02 a 0,03 na especificidade ficam dentro do ruído de
uma semente só (cerca de 1.200 pacientes sem a doença na validação). Os
finalistas são repetidos com outras sementes antes da decisão.

## Final

1. Avaliar o modelo escolhido uma única vez no teste (`RETINOPATIA_AVALIAR_TESTE = "1"`).
2. Escolher TTA2 ou TTA4 pela validação.
3. Exportar para TFLite (fp16 e int8) e conferir tamanho e paridade com o Keras.
4. Adaptar a API e o app para duas fotos por exame (tarefa separada).
