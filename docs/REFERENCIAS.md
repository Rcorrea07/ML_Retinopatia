# Referências

Fontes que orientam os experimentos do [plano v4](PLANO_V4.md). **Reabra o
original antes de implementar cada experimento**: este resumo não substitui o
texto nem o código dos autores. Se o original contradisser o resumo, corrija
aqui antes de treinar.

Catálogo completo levantado em 20/10/2026 (estado de acesso, commits, SHA-256):
`git show campanha-codex-2026-10-20:docs/referencias_retinopatia.json`.

## Métrica: por que os scores públicos não se comparam direto

| | Métrica | Conjunto |
|---|---|---|
| Kaggle EyePACS 2015 | Kappa quadrático nos **5 graus**, por olho | Teste da competição (53.576 imagens, hoje usadas no nosso treino) |
| Kaggle APTOS 2019 | Kappa quadrático nos 5 graus | Outro dataset (Índia), outra distribuição de graus |
| Este projeto | **Especificidade com sensibilidade 0,90**, 0 contra 1–4, por paciente; também ROC-AUC e AP | Validação e teste próprios (split por paciente) |

A cabeça de grau do projeto já fez kappa **0,81** no teste (B4 v2); o 1º lugar
de 2015 fez 0,85. O kappa da cabeça de grau é calculado em toda avaliação
(`avaliacao.kappa_grau`) só como ordem de grandeza: os conjuntos são outros.

## Kaggle EyePACS 2015 (diabetic-retinopathy-detection)

Leaderboards: [privado, final (F01)](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/leaderboard) ·
[público (F02)](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/leaderboard?tab=public).

| Posição | Equipe | Kappa privado | Código |
|---|---|---:|---|
| 1 | Min-Pooling (Ben Graham) | 0,84957 | F12, cópia local |
| 2 | o_O | 0,84478 | F13, cópia local |
| 3 | Reformed Gamblers | 0,83936 | ZIP anexo a F05, cópia local |
| 4 | Julian de Wit & Daniel Hammack | 0,83625 | não localizado |
| 5 | Jeffrey De Fauw | 0,82898 | F14, cópia local |

**F03 — [Min-Pooling / Ben Graham](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/writeups/min-pooling-competition-report-min-pooling-and-tha).**
Normaliza o raio, subtrai a média local de cor (desfoque gaussiano com sigma =
raio/30), mantém 90% do raio, rotações. Combina os olhos com Random Forest. É a
origem da nossa normalização de Ben, do recorte e da máscara de 90%.

**F04 — [o_O](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/writeups/o-o-team-o-o-solution-summary)** (código: F13).
Do `README.md` e do `solution_outline.md` da cópia local:
- recorte do menor retângulo que contém o olho; imagens de 128, 256 e 512 px;
- treino **progressivo**: a rede de 256 começa com os pesos da de 128, a de
  512 com os da de 256;
- regressão com erro quadrático; balanceamento que começa igual entre as
  classes e diminui a cada época até pesos 1 (grau 0) e 2 (demais);
- augmentation sempre ligado: rotação de 360°, translação, escala, cor;
- **blend por paciente**: as características dos dois olhos entram juntas
  num modelo final;
- só as redes 4x4 de 512 px já dão kappa **0,839**; o ensemble inteiro, 0,845.

**F05 — [Reformed Gamblers](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/writeups/reformed-gamblers-3rd-place-solution-report)**
([PDF v2](https://storage.googleapis.com/kaggle-forum-message-attachments/88866/2815/Team_Reformed_Gamblers_Solution_Summary_v2.pdf), código no ZIP do fórum).
Nove CNNs, regressão, augmentation na GPU, combinação dos olhos. Kappa de
validação **0,808 em 385 px contra 0,843 em 767 px**: a resolução foi a maior
diferença entre as redes deles.

**F15 — [Jeffrey De Fauw, 5º](https://defauw.ai/diabetic-retinopathy-detection/)** (código: F14).
Junta as representações dos dois olhos nas camadas densas, cuida de detalhes
finos (512 px) e reduz o oversampling até voltar à distribuição real.

## Kaggle APTOS 2019

**F06 — [Guanshuo Xu, 1º](https://www.kaggle.com/competitions/aptos2019-blindness-detection/writeups/guanshuo-xu-1st-place-solution-summary).**
Ensemble de oito modelos (EfficientNets, Inception, SE-ResNeXt), SmoothL1, GeM,
redimensionamento simples, dados de 2015 e pseudo-rótulos. Kappa privado de cerca
de 0,936 em outro dataset. Origem do GeM e da perda de Huber do projeto.

**F07 — [ODS.ai, 4º (2º após a limpeza)](https://www.kaggle.com/competitions/aptos2019-blindness-detection/writeups/ods-ai-eye-of-private-lb-4th-place-solution-2nd-af).**
EfficientNet B3–B5, pré-treino nos dados de 2015, TTA de espelhamento, sem ganho
com Ben no APTOS. A campanha de outubro mostrou o contrário no EyePACS (tirar Ben
custou 6 pontos de especificidade).

## Notebooks públicos (reabrir)

- **F08 — [Meena Vyas](https://www.kaggle.com/code/meenavyas/diabetic-retinopathy-detection).**
  O código não carregou em 20/10 (o Brave bloqueou o download). Reabrir.
- **F09 — [K Scott Mader, InceptionV3](https://www.kaggle.com/code/kmader/inceptionv3-for-retinopathy-gpu-hr).**
  O conteúdo não renderizou em 20/10. Reabrir; InceptionV3 não é candidata ao celular.

## Documentação técnica

- **F18 — [Destilação no Keras](https://keras.io/examples/vision/knowledge_distillation/)**
  ([código](https://github.com/keras-team/keras-io/blob/master/examples/vision/knowledge_distillation.py)).
  A mistura 0,7 rótulo / 0,3 professor do projeto é escolha própria, não do exemplo.
- **F16 — [FAQ do Colab](https://research.google.com/colaboratory/faq.html):** recursos e
  limites variam; em outubro o Colab aceitou 3 sessões com GPU, não 5.
- [EfficientNetV2 no Keras](https://keras.io/api/applications/efficientnet_v2/): entrada
  0–255 com o pré-processamento incluído.

## Cópias locais

Em `relatorios/referencias/2026-10-20/` (ignorada pelo git; detalhes em
`manifesto_downloads.json`). Leia sem executar (`unzip -p arquivo.zip caminho`).

| Arquivo | Fonte | Repositório e commit |
|---|---|---|
| `ben_graham_codigo.zip` | F12 | btgraham/SparseConvNet-archived, branch `kaggle_Diabetic_Retinopathy_competition`, `a6bdb0c` |
| `ben_graham_competitionreport.pdf` | F03 | relatório da competição |
| `o_o_codigo.zip` | F13 | sveitser/kaggle_diabetic, `e6f1cf3` |
| `jeffrey_codigo.zip` | F14 | JeffreyDF/kaggle_diabetic_retinopathy, `77c5781` |
| `rfg_submission_code.zip`, `reformed_gamblers_summary_v2.pdf` | F05 | anexos do fórum |

## O que reabrir antes de cada experimento

| Experimento | Fontes |
|---|---|
| R01 B0 512, R02 B0 640 | F04 (redes de 512), F05 (385 contra 767), F03 (sigma proporcional ao raio) |
| R03 destilação B4 → B0 | F18 |
| R04 pares de olhos | F15/F14 (representações juntas), F04 (blend por paciente) |
| R05 EfficientNetV2-B0 | F06, documentação da EfficientNetV2 |
| R06 progressivo ou treino longo | F04 (128 → 256 → 512, 250 épocas) |
