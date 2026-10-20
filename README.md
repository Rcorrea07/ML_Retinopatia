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

## Campanha de modelos leves (20/10/2026)

A branch `codex/campanha-retinopatia` acrescenta 11 perfis: dez variantes de
pré-processamento/treino e um classificador explícito dos cinco graus **0–4**.
EfficientNetB0 é a referência; MobileNetV3Large e destilação de B4 são alternativas.
A prioridade é PWA com inferência no servidor e preparação posterior para celular.

Abra os cinco notebooks `notebooks/Campanha_Posto_1.ipynb` a
`Campanha_Posto_5.ipynb`, cada um em uma sessão GPU independente. Eles executam
respectivamente `E01/E06/E11`, `E02/E07`, `E03/E08`, `E04/E09` e `E05/E10`.
Uma fila não implica GPU alocada; a disponibilidade depende do Colab.

Use o [guia operacional](docs/EXECUCAO_CAMPANHA.md) e o
[protocolo de comparação](docs/PLANO_EXPERIMENTOS_2026-10-20.md).
O [catálogo de fontes](docs/referencias_retinopatia.json) preserva todos os links
e downloads; reconsulte os relatórios e códigos durante as próximas alterações.
Treinamento exploratório avalia somente validação. O teste histórico já foi
consultado e não é uma nova avaliação cega.

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
    ├── avaliacao.py            # curvas, limiar, TTA, avaliação por olho e paciente
    └── diagnostico.py          # erros por grau, grau ≥ 2, agregação e bootstrap
```

## Organização no Google Drive

```
MyDrive/ProjetoRetinopatia/
├── modelos/          # modelos finais (.keras), ex.: B4_448_v3_1ccd2cb.keras
├── resultados/       # uma pasta por treino + registro_experimentos.csv
├── relatorios/       # relatórios em PDF
├── notebooks/        # cópias dos notebooks executados, com as saídas
└── dados_internos/   # splits (mesma divisão para todos os treinos) e cache das imagens
```

O nome do modelo junta rede, tamanho da imagem, versão do pré-processamento e
commit do código. O commit permite recuperar o código exato de cada treino.

## Pipeline

1. **Extração do dataset** — descompacta o zip do Drive no disco local do Colab
   (pula a etapa se os dados já estiverem lá). Se o zip das imagens já
   processadas desta versão existir em `ProjetoRetinopatia/dados_internos/cache/`, ele é
   restaurado e só o CSV de rótulos é extraído.
2. **Rótulos e divisão** — valida os nomes das imagens, cria a classe binária e
   divide em **treino, validação e teste** com `StratifiedGroupKFold` agrupado por
   paciente (10 dobras: uma para teste, uma para validação, oito para treino), para
   que os dois olhos da mesma pessoa nunca caiam em conjuntos diferentes. Os splits
   são salvos no Drive, versionados pelo esquema e pela semente, e reutilizados nas
   execuções seguintes.
3. **Pré-processamento offline (v3)** — feito uma única vez e gravado em cache
   (no disco do Colab e em zip no Drive): máscara da retina por limiar fixo,
   recorte quadrado centrado na retina (toda retina ocupa a mesma área da
   imagem), redimensionamento para 448×448, normalização de cor de Ben Graham
   calculada só dentro da retina e máscara circular a 90% do raio. Imagens cuja
   máscara cobre pouco da retina são marcadas como suspeitas.
4. **Pipelines `tf.data`** — só leitura, embaralhamento do treino e lotes;
   validação e teste sem embaralhamento. Nenhum aplica rescale: a EfficientNet
   espera imagens de 0 a 255.
5. **Modelo** — bloco de data augmentation na GPU (rotação de 360°,
   espelhamentos, zoom, brilho e contraste; só age no treino) + EfficientNetB4 +
   GeM pooling + duas saídas: `doente` (sigmoide, a decisão) e `grau` (regressão
   do grau 0 a 4 com perda de Huber, auxiliar). Precisão mista para acelerar o
   treino. O desbalanceamento é compensado com pesos por amostra.
6. **Treino em duas fases** — Fase 1 curta, só para aquecer as cabeças com a
   rede base congelada (`lr=1e-3`), e Fase 2 de fine-tuning com a base
   descongelada, mantendo as camadas de BatchNormalization congeladas, com taxa
   de aprendizado em cosseno com aquecimento (máximo `5e-5`). Pesos, histórico,
   predições e a configuração usada ficam em
   `ProjetoRetinopatia/resultados/<experimento>/` no Drive.
7. **Avaliação** — predições com test-time augmentation (espelhamentos). O limiar de
   decisão não é 0.5: ele é escolhido **na validação**, na curva ROC, como o ponto que
   atinge pelo menos **90% de sensibilidade** com o menor número de falsos positivos,
   e então aplicado ao **teste** no fluxo legado. Esse teste já foi consultado
   repetidamente e é agora um benchmark histórico; a campanha não o avalia. As métricas são
   reportadas por olho e por paciente (probabilidade do paciente = maior entre os dois
   olhos, como na regra clínica de encaminhamento).
8. **Diagnóstico** — sensibilidade por grau original, desempenho para retinopatia
   referenciável (grau ≥ 2), utilidade da cabeça de grau, formas de combinar os
   olhos e intervalos de confiança por bootstrap de pacientes.

## Origem das escolhas

As decisões da versão 2 vêm das soluções de topo das competições do Kaggle
[Diabetic Retinopathy Detection (2015)](https://www.kaggle.com/competitions/diabetic-retinopathy-detection)
e [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection):
normalização de raio e de cor de Ben Graham (1º lugar em 2015), imagens grandes,
rotações de 360°, combinação dos dois olhos do paciente (1º, 2º, 3º e 5º lugares em
2015), GeM pooling e perda de regressão SmoothL1/Huber (1º lugar APTOS), EfficientNet
com saídas de regressão/ordinais e TTA com espelhamentos (notebooks de maior
pontuação da APTOS).

Os 53.576 rótulos do teste original de 2015 já estão disponíveis para treino.
APTOS não entra no treino desta campanha e pode servir a uma avaliação externa,
após auditoria de origem e duplicatas. A comparação inicial usa um único modelo
e uma passagem por imagem; TTA é uma análise secundária.

## Reprodutibilidade

Sementes separadas: `SEED_SPLIT=42` mantém os pacientes originais e
`SEED_TREINO` controla Python, NumPy, TensorFlow e embaralhamento. Trocar apenas
a semente de treino não altera o split nem o cache. A campanha identifica
o cache por hash da receita de imagem; cada execução tem perfil, hash de
configuração, commit e UUID próprios, preservando os resultados anteriores.

Para carregar o modelo salvo, informe a camada personalizada:

```python
from retinopatia.modelo import GeM
modelo = tf.keras.models.load_model(
    configuracao.CAMINHO_MODELO_FINAL,
    custom_objects={"GeM": GeM}
)
```
# ml-retinopatia
