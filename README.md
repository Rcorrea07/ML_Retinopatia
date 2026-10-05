# Projeto Retinopatia

Classificador binário de **retinopatia diabética** em fotos de fundo de olho
(retinografias), usando transfer learning com **EfficientNet** (Keras/TensorFlow).

O grau original de severidade do dataset (0 a 4) é colapsado em duas classes para a
decisão do modelo; o grau completo continua sendo usado por uma cabeça auxiliar de
regressão:

| Classe | Significado |
|---|---|
| 0 | Sem retinopatia (grau 0) |
| 1 | Com retinopatia (graus 1 a 4) |

## Como executar

O projeto é **executado no Google Colab** (por causa da GPU; a configuração padrão,
EfficientNetB4 em 448×448, foi pensada para L4 ou A100) e **editado localmente**
no VS Code. O código Python vive neste repositório e é trazido para o Colab por
`git clone`/`git pull`.

1. Abra `notebooks/Retinopatia_0_1.ipynb` no Google Colab.
2. Ative uma GPU: *Ambiente de execução → Alterar o tipo de ambiente de execução → GPU*.
3. Execute as células de cima para baixo.

A primeira célula clona (ou atualiza) este repositório dentro do Colab e coloca
`src/` no `sys.path`, tornando o pacote `retinopatia` importável. O `autoreload`
está ativado, então basta um `git pull` para que edições feitas no VS Code passem
a valer na sessão que já está aberta.

O dataset é lido do Google Drive, a partir de:

```
/content/drive/MyDrive/Datasets/diabetic-retinopathy-detection.zip
```

Não é necessário instalar nada localmente para editar o código: o `requirements.txt`
serve apenas como documentação das bibliotecas usadas no Colab.

## Estrutura

```
├── notebooks/
│   └── Retinopatia_0_1.ipynb   # orquestra a execução no Colab
└── src/retinopatia/
    ├── configuracao.py         # sementes, caminhos e hiperparâmetros
    ├── utilitarios.py          # sementes, determinismo e precisão mista
    ├── ambiente_colab.py       # Drive, extração do dataset e download
    ├── conjunto_dados.py       # CSV, rótulos e divisão treino/validação/teste
    ├── pre_processamento.py    # centralização da retina, cor de Ben Graham e máscara
    ├── pipeline_dados.py       # pipelines tf.data e data augmentation
    ├── modelo.py               # EfficientNet + GeM + cabeças binária e de grau
    ├── treinamento.py          # pesos de classe, callbacks e as duas fases
    └── avaliacao.py            # curvas, limiar, TTA, avaliação por olho e paciente
```

## Pipeline

1. **Extração do dataset** — descompacta o zip do Drive no disco local do Colab
   (pula a etapa se os dados já estiverem lá).
2. **Rótulos e divisão** — valida os nomes das imagens, cria a classe binária e
   divide em **treino, validação e teste** com `StratifiedGroupKFold` agrupado por
   paciente (10 dobras: uma para teste, uma para validação, oito para treino), para
   que os dois olhos da mesma pessoa nunca caiam em conjuntos diferentes. Os splits
   são salvos no Drive, versionados pelo esquema e pela semente, e reutilizados nas
   execuções seguintes.
3. **Pré-processamento offline** — feito uma única vez e gravado em cache:
   máscara de Otsu, recorte quadrado centrado na retina (toda retina ocupa a mesma
   área da imagem), redimensionamento para 448×448, normalização de cor de Ben
   Graham e máscara circular a 90% do raio.
4. **Pipelines `tf.data`** — data augmentation no treino (rotação de 360°,
   espelhamentos, zoom, brilho e contraste); validação e teste limpos e sem
   embaralhamento. Nenhum aplica rescale: a EfficientNet espera imagens de 0 a 255.
5. **Modelo** — EfficientNetB4 + GeM pooling + duas saídas: `doente` (sigmoide, a
   decisão) e `grau` (regressão do grau 0 a 4 com perda de Huber, auxiliar).
   Precisão mista para acelerar o treino. O desbalanceamento é compensado com pesos
   por amostra.
6. **Treino em duas fases** — Fase 1 com a rede base congelada (`lr=1e-3`) e Fase 2
   de fine-tuning com a base descongelada, mantendo as camadas de BatchNormalization
   congeladas (`lr=5e-5`).
7. **Avaliação** — predições com test-time augmentation (espelhamentos). O limiar de
   decisão não é 0.5: ele é escolhido **na validação**, na curva ROC, como o ponto que
   atinge pelo menos **90% de sensibilidade** com o menor número de falsos positivos,
   e então aplicado ao **teste**, que dá a estimativa honesta. As métricas são
   reportadas por olho e por paciente (probabilidade do paciente = maior entre os dois
   olhos, como na regra clínica de encaminhamento).

## Origem das escolhas

As decisões da versão 2 vêm das soluções de topo das competições do Kaggle
[Diabetic Retinopathy Detection (2015)](https://www.kaggle.com/competitions/diabetic-retinopathy-detection)
e [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection):
normalização de raio e de cor de Ben Graham (1º lugar em 2015), imagens grandes,
rotações de 360°, combinação dos dois olhos do paciente (1º, 2º, 3º e 5º lugares em
2015), GeM pooling e perda de regressão SmoothL1/Huber (1º lugar APTOS), EfficientNet
com saídas de regressão/ordinais e TTA com espelhamentos (notebooks de maior
pontuação da APTOS).

Próximos passos possíveis: incluir os rótulos do teste de 2015 (~53 mil imagens) e
os dados da APTOS 2019, e fazer ensemble de dobras ou sementes.

## Reprodutibilidade

Semente fixa (`SEED = 42`) aplicada a Python, NumPy e TensorFlow, com operações
determinísticas ativadas quando o ambiente permite. Trocar a semente, o
pré-processamento ou o mapeamento dos rótulos invalida os caches de imagens
processadas e os CSVs de split já salvos — mude `VERSAO_PRE_PROCESSAMENTO` ou
`VERSAO_SPLIT` em `configuracao.py`, ou regenere-os.

Para carregar o modelo salvo, informe a camada personalizada:

```python
from retinopatia.modelo import GeM
modelo = tf.keras.models.load_model(
    "modelo_retinopatia_final.keras",
    custom_objects={"GeM": GeM}
)
```
