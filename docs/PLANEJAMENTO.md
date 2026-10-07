# Planejamento das próximas etapas

Atualizado em 07/10/2026, depois dos treinos v3 (B4 e B0, commit `1ccd2cb`).
Base: relatórios de 06/10 (treino v2 de 10 h) e 07/10 (B4 × B0), contexto do
PWA (`contexto_codex_retinopatia_pwa.txt`) e registro da integração do modelo
no app (23/07).

## 1. Onde o projeto está

O objetivo é um PWA de apoio à triagem: o médico envia a foto do fundo de olho,
a API de ML (FastAPI) devolve a sugestão e o resultado fica salvo no Supabase.
O app já funciona de ponta a ponta com o modelo antigo de julho
(EfficientNetB0, 224 px, recorte de Otsu, limiar 0,2476).

Desde então, o modelo evoluiu bastante, mas **o app ainda não usa nenhum dos
modelos novos**.

| Modelo | ROC-AUC paciente (teste) | Especificidade paciente com sens. 0,90 | Treino | Tamanho |
|---|---|---|---|---|
| B0 224 (julho, no app) | não medido no teste atual | — | — | — |
| B4 448 v2 | 0,915 | 0,70 | ~10 h | 17,7 M parâmetros |
| B4 448 v3 | 0,905 | 0,63 | ~53 min | 17,7 M |
| B0 448 v3 | 0,884 | 0,58 | ~29 min | 4,1 M |
| Média B4 v3 + B4 v2 | 0,915 | 0,72 | — | 2 × 17,7 M |

## 2. O que deu certo

- **Divisão por paciente e regra validação → teste:** os números do teste batem
  com os da validação em todos os treinos. Não há vazamento.
- **Velocidade:** o treino caiu de ~10 h para ~1 h com o augmentation na GPU.
  Experimentos agora são baratos.
- **Rastreabilidade:** cada treino tem commit, configuração, histórico e
  predições salvos. O Drive está organizado (`modelos/`, `resultados/`,
  `relatorios/`).
- **Detecção dos casos graves:** graus 3 e 4 são detectados em 99–100% dos
  olhos. No critério clínico de encaminhamento (grau ≥ 2), a ROC-AUC por
  paciente chega a 0,94–0,95.
- **Ensemble:** juntar dois B4 com pré-processamentos diferentes deu o melhor
  resultado, sem treino extra.

## 3. O que deu errado ou ficou em aberto

1. **Teto de acurácia.** Todos os modelos param em ~0,91 de ROC-AUC. Os erros
   se concentram no grau 1 (~23% dos olhos de grau 1 passam), o rótulo mais
   ruidoso do dataset. Mudar treino não move esse teto; mais dados devem mover.
2. **Especificidade por paciente da B4 v3 caiu** (0,70 → 0,63). A queda vem da
   regra que junta os dois olhos pelo máximo; com a média, a v3 melhora. Não se
   sabe ainda se é efeito do pré-processamento v3 ou variação entre treinos.
3. **Muitos falsos positivos:** com 90% de sensibilidade, ~37% dos olhos
   saudáveis são sinalizados. Aceitável para triagem, mas é o ponto fraco.
4. **O app usa um modelo antigo**, com outro pré-processamento e outra
   resolução. Trocar o modelo exige trocar também o pré-processamento da API.
5. **O app espera grau 0–4** no contrato (`predicted_class`, `label`), mas o
   modelo decide só "tem ou não tem retinopatia". A cabeça de grau existe, mas
   não é usada na saída.
6. **Implantação no Raspberry Pi** ainda não foi avaliada (seção 5).
7. (Corrigido neste commit) A lista de máscaras suspeitas se perdia quando o
   pré-processamento falhava.

## 4. Etapas propostas

Cada etapa compara modelos **só na validação**; o teste é medido uma vez, no
modelo escolhido.

### Etapa A — Medir a variação entre treinos (1 dia)

- Treinar a B4 v3 com mais 2 sementes (~1 h cada, em paralelo).
- Responde se a queda da especificidade (item 3.2) é real.
- Os três modelos já servem para o ensemble da Etapa B.
- Pré-requisito no código: a semente vir da primeira célula do notebook, como
  o backbone.

### Etapa B — Ensemble e agregação no código (1–2 dias)

- Função em `avaliacao` que faz a média de vários modelos e escolhe, na
  validação, a combinação e a forma de juntar os olhos (máximo ou média).
- Meta: especificidade por paciente ≥ 0,72 com sensibilidade 0,90.

### Etapa C — Mais dados (2–4 dias)

- Incluir os rótulos públicos do conjunto de teste do Kaggle 2015
  (~53 mil imagens), mais que dobrando o treino.
- Opcional: APTOS 2019 (~3,7 mil imagens de outra câmera), bom para medir
  generalização.
- Mantém o teste atual intocado para comparar com os números de hoje.
- É a etapa com mais chance de passar do teto de 0,91.

### Etapa D — Modelo leve para o Raspberry Pi (2–3 dias)

Ver seção 5. Resumo: treinar grande, implantar pequeno.

- Destilação: o ensemble B4 (professor) ensina uma B0 (aluno), que imita as
  probabilidades do professor. Costuma recuperar boa parte da diferença
  B0 × B4.
- Exportar para TensorFlow Lite (float16 e int8) e medir no Pi.

### Etapa E — Atualizar o app (2–3 dias)

- Trocar o modelo na `ml-api` pelo escolhido nas etapas anteriores.
- Portar o pré-processamento v3 (`processar_retina`) para a API, sem copiar:
  a API deve importar a mesma função ou um pacote extraído deste repositório.
- Atualizar limiar e regra por paciente quando os dois olhos forem enviados.
- Devolver também o grau provável (cabeça de grau, kappa 0,79) e a
  probabilidade, cumprindo o contrato `predicted_class` / `label` /
  `confidence`.
- Registrar no Supabase o nome do modelo (ex.: `B0_448_v3_1ccd2cb`).

## 5. B4 × B0 no Raspberry Pi

### Tamanho e custo

| | EfficientNetB0 448 | EfficientNetB4 448 |
|---|---|---|
| Parâmetros | 4,1 M | 17,7 M |
| Operações por imagem (estimativa) | ~1,6 bilhão | ~5,8 bilhões |
| Pesos float32 / float16 / int8 | ~16 / 8 / 4 MB | ~71 / 35 / 18 MB |
| Arquivo `.keras` atual | 77 MB | 335 MB |

O `.keras` é maior porque guarda também o estado do otimizador. Para implantar,
basta exportar os pesos (TFLite).

### Tempo esperado (estimativa, a medir)

| | Pi 5 (8 GB) | Pi 4 |
|---|---|---|
| B0 448, uma imagem | ~0,2–0,5 s | ~0,6–1,5 s |
| B4 448, uma imagem | ~0,5–2 s | ~2–6 s |
| B4 448 com TTA (4 variantes) | ~2–8 s | ~8–24 s |
| Ensemble de 2 B4 com TTA | ~4–16 s | inviável na prática |

São faixas estimadas a partir do número de operações, não medições. O
pré-processamento da foto original (~3000 px, OpenCV) soma ~0,3–1 s.

### Leitura

- **A B4 roda no Pi 5.** Para triagem (um exame por vez, que leva minutos), alguns
  segundos são aceitáveis. Memória não é problema.
- **O problema é o ensemble e a TTA**, que multiplicam o tempo. No Pi 4 isso fica
  lento demais.
- **A B0 é o candidato natural para o Pi**, mas sozinha perde ~0,02 de ROC-AUC e
  ~7 pontos de especificidade. A destilação (Etapa D) é a forma de fechar essa
  diferença.
- **Acelerador (opcional):** o AI HAT+ do Pi 5 (Hailo) exige modelo int8
  compilado e nem todas as camadas da EfficientNet são suportadas. A variante
  EfficientNet-Lite foi feita para isso. Só vale considerar se a CPU não bastar.

### Critério de decisão

Escolher o menor modelo que, no Pi alvo:

- fique a no máximo 0,01 de ROC-AUC por paciente do melhor modelo (validação);
- responda em até ~5 s por exame (dois olhos), incluindo o pré-processamento.

## 6. Perguntas em aberto

1. **Qual Raspberry Pi** (4 ou 5, quanta RAM, tem acelerador)?
2. **Qual o papel do Pi:** ele roda a `ml-api` como servidor da clínica, ou o
   modelo precisa funcionar sem internet, dentro de um aparelho?
3. **A interface precisa mostrar o grau 0–4**, ou basta "encaminhar / não
   encaminhar"? Isso decide se o grau vira saída oficial do modelo.

## 7. Ordem recomendada

1. Etapa A (variação) e Etapa C (dados) podem começar juntas.
2. Etapa B depois de A.
3. Etapa D depois de B e C (o professor precisa ser o melhor modelo disponível).
4. Etapa E por último, com o modelo final, ou antes, se o app precisar de um
   modelo melhor já: a B4 v3 sozinha pode entrar no app como versão
   intermediária.
