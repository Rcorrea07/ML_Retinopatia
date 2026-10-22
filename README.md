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

O projeto é **executado no Google Colab** (GPU A100 com RAM alta) e **editado
localmente** no VS Code. O código Python vive neste repositório e é trazido para o
Colab por `git clone`/`git pull`.

1. Abra `notebooks/Retinopatia_0_1.ipynb` no Google Colab (ele já pede A100 com RAM alta).
2. Na primeira célula, escolha o experimento (`EXPERIMENTO`, lista em
   `src/retinopatia/experimentos.py`).
3. Execute as células de cima para baixo.

Um experimento por sessão. Para rodar vários em paralelo, use uma cópia do
notebook por sessão, mudando só `EXPERIMENTO`. O passo a passo das rodadas, com a
tabela de resultados, está em [`docs/PLANO_V4.md`](docs/PLANO_V4.md); as fontes, em
[`docs/REFERENCIAS.md`](docs/REFERENCIAS.md).

Os experimentos são comparados só na validação. O teste fica fechado até o modelo
final (`RETINOPATIA_AVALIAR_TESTE = "1"` na primeira célula).

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
├── docs/                       # plano v4, referências e planejamento anterior
├── notebooks/
│   ├── Retinopatia_0_1.ipynb   # treino: orquestra a execução no Colab
│   ├── Preparar_Caches.ipynb   # caches de imagens (conta pessoal), uma vez por tamanho
│   └── Analisar_Resultados.ipynb  # comparação das execuções salvas (só CPU)
├── tests/                      # testes sem TensorFlow
└── src/retinopatia/
    ├── experimentos.py         # experimentos nomeados (o que cada um muda)
    ├── configuracao.py         # sementes, caminhos e hiperparâmetros
    ├── utilitarios.py          # sementes, determinismo e precisão mista
    ├── ambiente_colab.py       # Drive, extração do dataset e download
    ├── conjunto_dados.py       # CSV, rótulos e divisão treino/validação/teste
    ├── pre_processamento.py    # centralização da retina, cor de Ben Graham e máscara
    ├── pipeline_dados.py       # pipelines tf.data e data augmentation
    ├── modelo.py               # EfficientNet + GeM + cabeças binária e de grau
    ├── treinamento.py          # pesos de classe, callbacks e as duas fases
    ├── avaliacao.py            # curvas, limiar, TTA, avaliação por olho e paciente
    ├── diagnostico.py          # erros por grau, grau ≥ 2, agregação e bootstrap
    └── analise.py              # análises das predições salvas, sem TensorFlow
```

## Organização no Google Drive

```
MyDrive/ProjetoRetinopatia/
├── modelos/          # modelos finais (.keras), um por execução
├── resultados/       # uma pasta por execução (resumo.csv, predições, históricos…)
│                     # + registro_experimentos.csv (consolidado)
├── relatorios/       # relatórios em PDF
├── notebooks/        # cópias dos notebooks executados, com as saídas
└── dados_internos/   # splits (mesma divisão para todos os treinos), caches de 448
                      # e probabilidades do professor da destilação

MyDrive/ProjetoRetinopatia_caches/   # atalho para a pasta da conta pessoal:
                                     # caches de 512 px ou mais
```

O nome do modelo e da pasta de resultados junta experimento, rede, tamanho da
imagem, versão do pré-processamento, commit do código e data-hora da execução:
nada é sobrescrito, e o commit permite recuperar o código exato de cada treino.

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
   e então aplicado ao **teste**, que dá a estimativa honesta. As métricas são
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

Próximos passos possíveis: incluir os rótulos do teste de 2015 (~53 mil imagens) e
os dados da APTOS 2019, e fazer ensemble de dobras ou sementes.

## Reprodutibilidade

A divisão treino/validação/teste usa sempre `SEED_SPLIT = 42`; a semente do treino
(`SEED_TREINO`, padrão 42, `RETINOPATIA_SEED_TREINO` para repetir um experimento)
vale para Python, NumPy e TensorFlow, com operações determinísticas ativadas quando
o ambiente permite. Mudar o pré-processamento ou o mapeamento dos rótulos invalida
os caches de imagens processadas e os CSVs de split já salvos — mude
`VERSAO_PRE_PROCESSAMENTO` ou `VERSAO_SPLIT` em `configuracao.py`, ou regenere-os.

Para carregar o modelo salvo, informe a camada personalizada:

```python
from retinopatia.modelo import carregar_modelo_salvo
modelo = carregar_modelo_salvo(configuracao.CAMINHO_MODELO_FINAL)
```
# ml-retinopatia
