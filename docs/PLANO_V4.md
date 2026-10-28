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
| R04 | 2 | B0 com pares de olhos, 640 px (pulado: o combinador não mostrou ganho) | — | — | — | — | — |
| R05 | 2 | EfficientNetV2-B0, 640 px | 0,744 | 0,740 | 0,781 | 0,929 | 0,838 |
| R06 | 2 | B0 640 com treino mais longo (18 épocas na Fase 2) | 0,755 | 0,723 | 0,715 | 0,926 | 0,830 |
| R07 | 3 | R02 (B0 640) com semente 17 (mesmo split) | 0,753 | 0,759 | 0,738 | 0,924 | 0,835 |
| R08 | 3 | R05 (V2-B0 640) com semente 17 (mesmo split) | 0,771 | 0,756 | 0,785 | 0,928 | 0,838 |
| R09 | final | Modelo escolhido: teste (uma vez) e TFLite | | | | | |
| R10 | 3 | B0 em 768 px (o grau 1 não melhorou de 448 para 640) | 0,757 | 0,790 | 0,780 | 0,927 | 0,823 |
| R10 s17 | 4 | R10 (B0 768) com semente 17 | 0,747 | 0,785 | 0,765 | 0,926 | 0,837 |
| R11 | 4 | V2-B0 em 768 px, semente 42 | 0,775 | 0,747 | 0,757 | 0,928 | 0,843 |
| R11 s17 | 4 | V2-B0 em 768 px, semente 17 | 0,769 | 0,765 | 0,757 | 0,931 | 0,843 |

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

### Antes da leva 2

- **Combinar os dois olhos depois do treino não ajudou.** Uma regressão
  logística sobre os dois olhos (probabilidades e graus), em validação cruzada
  por paciente, deu especificidade 0,737 no R02 contra 0,732 da regra do máximo
  (ROC 0,922 contra 0,923); no E01, 0,660 contra 0,671. Misturar as cabeças
  `doente` e `grau` também empatou (0,736). O R04 (pares de olhos, cerca de o
  dobro do custo) fica de fora enquanto o orçamento de GPU estiver curto.
- **O cache de 640 no Drive pessoal não foi possível.** O Colab da conta do
  projeto recusa montar o Drive de outra conta ("mount failed"), e o Colab
  gratuito da conta pessoal não tem disco para extrair o dataset bruto (~105 GB).
  R05 e R06 processam as imagens de 640 na própria sessão (cerca de 1 h a mais).

### Leva 2 (22/10/2026, commit `1b45ca0`, R05 e R06 em paralelo)

- **R05 (EfficientNetV2-B0, 640 px) empatou com o R02, com leve vantagem.**
  Sem TTA, ROC-AUC 0,929 e especificidade 0,744 (R02: 0,923 e 0,732). Com
  TTA4, 0,932 e 0,781 (R02: 0,926 e 0,714). Kappa da cabeça de grau 0,838
  (R02: 0,818). A ROC-AUC sobe cerca de 0,006 em todos os modos; o salto da
  especificidade com TTA4 está no ponto de 90% de sensibilidade, que oscila de
  um modo para outro, e precisa de outras sementes antes de valer.
- **Por olho, o R05 é igual ao R02:** ROC-AUC 0,932 com TTA4 (R02: 0,931); grau
  0 marcado como doente em 25,6% dos olhos (R02: 25,5%), grau 1 detectado em
  69% (R02: 69%), graus 2 a 4 em 96–99%. A pequena vantagem aparece só por
  paciente.
- **Treina mais rápido:** 470 s por época na Fase 2 contra 656 s da B0 a 640
  (−28%), com pico de 6,1 GB de GPU. A melhor época foi a última (12 de 12),
  mas a PR-AUC de validação já tinha estabilizado (0,8887 → 0,8892 nas três
  últimas).
- **No celular, custa mais que a B0:** pelo artigo da EfficientNetV2, a V2-B0
  tem 7,1 M de parâmetros e cerca de 0,72 GFLOPs a 224 px, contra 5,3 M e
  0,39 GFLOPs da B0. Medir a latência no TFLite antes de escolhê-la.
- **R06 (B0 640, 18 épocas) não ganhou com o treino mais longo.** A melhor
  época foi a 10 (PR-AUC de validação 0,8875) e o early stopping parou na 14.
  Por paciente: ROC-AUC 0,926 e especificidade 0,755 sem TTA; 0,929 e 0,715
  com TTA4; kappa 0,830. Por olho, igual ao R02 (grau 0 marcado em 26,6% dos
  olhos, grau 1 detectado em 70%). No R02, a melhor época ser a última vinha do
  fim do cosseno (taxa de aprendizado perto de zero), e não de falta de épocas:
  as 12 épocas ficam.
- **Conclusão da leva 2:** R02, R05 e R06 ficam a menos de 0,006 de ROC-AUC
  entre si, e as especificidades oscilam dentro do ruído. O R05 fica à frente na
  ROC-AUC em todos os modos e no kappa. A única alavanca clara continua sendo
  os 640 px. Nenhuma execução mudou o ponto de operação por olho: o grau 1, em
  cerca de 70%, é o gargalo.
- **As unidades de computação do Colab acabaram** no fim da leva 2 (o painel de
  recursos diz que não há unidades disponíveis). As levas 1 e 2 gastaram as 156
  do Pro+. As próximas rodadas dependem de unidades novas.
- **Próxima rodada, quando houver unidades:** a escolha entre B0 e V2-B0 pede
  uma segunda semente de cada (R02 e R05 com `RETINOPATIA_SEED_TREINO=17`),
  porque a diferença entre elas é menor que o ruído. Depois, o teste do modelo
  escolhido e a latência no TFLite.

### Rodada 3 (27/10/2026, commit `c35c12a`, R07, R08 e R10 em paralelo)

Os notebooks estão em `notebooks/execucoes/` (R07, R08 e R10). Para o Drive
montar, rode a cópia salva no Drive ("Arquivo → Salvar uma cópia no Drive"),
não o notebook aberto direto do GitHub. A rodada gastou cerca de 105 das 600
unidades (sobraram 495).

Validação por paciente; médias das duas sementes (42 e 17):

| Configuração | ROC sem TTA | Espec. sem TTA | ROC TTA4 | Espec. TTA4 | Kappa grau |
|---|---:|---:|---:|---:|---:|
| B0 640 (R02 + R07) | 0,924 | 0,742 | 0,927 | 0,726 | 0,826 |
| V2-B0 640 (R05 + R08) | 0,929 | 0,757 | 0,931 | 0,783 | 0,838 |
| B0 768 (R10, uma semente) | 0,927 | 0,757 | 0,930 | 0,780 | 0,823 |

- **A V2-B0 vence a B0 em 640 px, nas duas sementes.** A ROC-AUC sem TTA das
  duas V2-B0 (0,929 e 0,928) fica acima das duas B0 (0,923 e 0,924), e com
  TTA4 a especificidade das V2-B0 (0,781 e 0,785) fica acima das B0 (0,714 e
  0,738). Kappa 0,838 contra 0,826. A vantagem só aparece por paciente: por
  olho, as duas empatam (ROC-AUC 0,932 e 0,931 com TTA4).
- **Ruído de semente medido:** a mesma receita variou de 2 a 2,6 pontos de
  especificidade sem TTA (B0: 0,732 e 0,753; V2-B0: 0,744 e 0,771).
- **768 px (R10) é o melhor por olho:** ROC-AUC 0,937 e especificidade 0,782
  com TTA4 (B0 640: 0,931 a 0,932 e 0,745 a 0,768). Os olhos sadios marcados
  como doentes caíram para 21,8% (B0 640: 23 a 26%). Por paciente, empata com
  a V2-B0 640. Custou 942 s por época na Fase 2 (B0 640: 656 s), pico de
  12,6 GB de GPU e 25 GB de RAM, cerca de 5 h 20 min no total. Melhor época
  11 de 12, com a maior PR-AUC de validação até agora (0,897).
- **O grau 1 não se move:** 68% a 70% dos olhos em todas as execuções da
  rodada, com B0 ou V2-B0, em 640 ou 768 px (69% a 77% nos treinos anteriores
  medidos, desde 448 px). Resolução e rede não são a alavanca para ele; pode ser o limite dos
  próprios rótulos do grau 1.
- **Custo no celular, relativo à B0 640:** B0 768 ≈ 1,4× (pixels); V2-B0 640
  ≈ 1,8× (FLOPs do artigo); V2-B0 768 ≈ 2,7×.

Próximo passo sugerido: R11 = V2-B0 em 768 px com as sementes 42 e 17, e o
R10 com a semente 17. Assim os três candidatos (V2-B0 640, B0 768 e V2-B0 768)
ficam com duas sementes cada. Depois, R09: teste uma única vez no escolhido e
TFLite (fp16 e int8), conferindo tamanho, latência e paridade com o Keras.

### Antes da rodada 4: o custo

A rodada 3 gastou ~105 unidades (A100 a ~6,8 unidades/h por sessão, 3
sessões), mas só ~70 foram de treino. Cerca de 20 foram ~1 h por sessão
processando as mesmas imagens com a GPU parada, e de 10 a 14 foram sessões
ligadas depois do fim (o R07 ficou ~1,5 h parado). Cada treino em 768 px usou
~13 dos 80 GB de VRAM e ~25 dos 167 GB de RAM da A100.

O Colab não deixa escolher a A100 de 40 GB (oferece CPU, T4, L4, A100, H100,
G4 e TPUs). T4 e L4 têm memória suficiente, mas calculam bem mais devagar (na
campanha de outubro, a T4 levou ~5 h onde a A100 leva ~1,5 h): mais baratas
por hora, não necessariamente por treino, e um treino em 768 px numa T4
chegaria perto do limite de 24 h.

Por isso a rodada 4 roda numa sessão só (`notebooks/execucoes/Rodada4_768.ipynb`,
lógica em `src/retinopatia/execucao.py`): as imagens são processadas uma vez,
os três treinos rodam ao mesmo tempo na mesma A100 (cada um num processo, com
a VRAM alocada sob demanda), cada um passa antes por uma fumaça de poucos
minutos, e a sessão desliga sozinha no fim, também se der erro. Os
checkpoints de cada fase ficam no disco local (o `.keras` final já tem os
melhores pesos). Estimativa: de 45 a 60 unidades, contra ~100 com três
sessões, dependendo de quanto os três treinos disputam a GPU.

### Rodada 4 (27–28/10/2026, commit `95e73a9`, uma sessão, 3 treinos juntos)

R10 com a semente 17 e R11 (V2-B0 768) com as sementes 42 e 17, ao mesmo
tempo na mesma A100. Validação por paciente; médias das sementes 42 e 17:

| Configuração | ROC sem TTA | Espec. sem TTA | ROC TTA4 | Espec. TTA4 | Kappa | Por olho, TTA4: ROC · espec. | Custo no celular |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 640 | 0,924 | 0,742 | 0,927 | 0,726 | 0,826 | 0,932 · 0,756 | 1× |
| V2-B0 640 | 0,929 | 0,757 | 0,931 | 0,783 | 0,838 | 0,932 · 0,749 | ≈ 1,8× |
| **B0 768** | 0,927 | 0,752 | 0,931 | 0,772 | 0,830 | **0,938 · 0,782** | ≈ 1,4× |
| V2-B0 768 | 0,929 | 0,772 | 0,929 | 0,757 | 0,843 | 0,935 · 0,753 | ≈ 2,7× |

- **Por paciente, os quatro empatam:** a ROC-AUC fica entre 0,924 e 0,931, e
  a especificidade troca de lugar conforme o modo de TTA (diferenças de 1 a 5
  pontos, dentro do ruído de semente).
- **A V2-B0 em 768 px não melhora a de 640 px** (ROC 0,929 nas duas, kappa
  0,843 contra 0,838) e custa ~1,5× mais no celular. Fica de fora.
- **O B0 768 é o melhor por olho nas duas sementes:** ROC-AUC 0,937 e 0,938,
  especificidade 0,782 e 0,783, com 21,7 a 21,8% dos olhos sadios marcados
  como doentes (as outras configurações: 23 a 26%). É também o segundo mais
  barato no celular.
- **O grau 1 continua parado:** 68 a 69% nas três execuções; graus 2 a 4 em
  96 a 100%.
- **Escolha sugerida para o celular: B0 768.** Reserva: V2-B0 640. Próximo
  passo (R09): avaliar o B0 768 no teste uma única vez e exportar para TFLite
  (fp16 e int8), medindo tamanho, latência no celular e paridade com o Keras.

**Custo:** ~65 unidades (saldo de 495 para 430), contra ~105 na rodada 3,
que teve treinos mais leves (dois em 640 px). A sessão durou ~9,6 h: ~2 h
processando as imagens de 768 px uma vez, a fumaça dos três (~3 min) e os
treinos. Com os três juntos, a GPU ficou em 100% e usou 49 de 80 GB; cada
época da Fase 2 levou ~31 min no V2-B0 e ~46 min no B0 (sozinhos, ~11 e
~16 min). Rodar junto rendeu pouco, cerca de 10% a mais de trabalho por hora.
A economia veio de processar as imagens uma vez e de a sessão desligar
sozinha ao terminar.

Diferenças de cerca de 0,02 a 0,03 na especificidade ficam dentro do ruído de
uma semente só (cerca de 1.200 pacientes sem a doença na validação). Os
finalistas são repetidos com outras sementes antes da decisão.

## Final

1. Avaliar o modelo escolhido uma única vez no teste (`RETINOPATIA_AVALIAR_TESTE = "1"`).
2. Escolher TTA2 ou TTA4 pela validação.
3. Exportar para TFLite (fp16 e int8) e conferir tamanho e paridade com o Keras.
4. Adaptar a API e o app para duas fotos por exame (tarefa separada).
