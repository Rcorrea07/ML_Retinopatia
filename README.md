# Projeto Retinopatia

Classificador binário de **retinopatia diabética** em fotos de fundo de olho
(retinografias), usando transfer learning com **EfficientNetB0** (Keras/TensorFlow).

O grau original de severidade do dataset (0 a 4) é colapsado em duas classes:

| Classe | Significado |
|---|---|
| 0 | Sem retinopatia (grau 0) |
| 1 | Com retinopatia (graus 1 a 4) |

## Como executar

O projeto é **executado no Google Colab** (por causa da GPU) e **editado localmente**
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
    ├── utilitarios.py          # sementes e determinismo
    ├── ambiente_colab.py       # Drive, extração do dataset e download
    ├── conjunto_dados.py       # CSV, rótulos e divisão treino/validação
    ├── pre_processamento.py    # recorte da retina pelo método de Otsu
    ├── pipeline_dados.py       # geradores de imagens do Keras
    ├── modelo.py               # EfficientNetB0 + cabeça binária
    ├── treinamento.py          # pesos de classe, callbacks e as duas fases
    └── avaliacao.py            # curvas, limiar de decisão e matriz de confusão
```

## Pipeline

1. **Extração do dataset** — descompacta o zip do Drive no disco local do Colab
   (pula a etapa se os dados já estiverem lá).
2. **Rótulos e divisão** — valida os nomes das imagens, cria a classe binária e
   divide treino/validação com `StratifiedGroupKFold` agrupado por paciente, para
   que os dois olhos da mesma pessoa nunca caiam em conjuntos diferentes. Os splits
   são salvos no Drive, versionados pela semente.
3. **Pré-processamento offline** — recorta a retina com o método de Otsu e
   redimensiona para 224×224 uma única vez, gravando o resultado em cache
   (evita pagar o custo do recorte a cada época).
4. **Geradores** — data augmentation no treino; validação limpa e sem embaralhamento.
   Nenhum dos dois aplica rescale: a EfficientNet espera imagens de 0 a 255.
5. **Treino em duas fases** — Fase 1 com a rede base congelada (`lr=1e-3`) e Fase 2
   de fine-tuning com a base descongelada, mantendo as camadas de BatchNormalization
   congeladas (`lr=1e-5`).
6. **Avaliação** — o limiar de decisão não é 0.5: ele é escolhido na curva ROC como o
   ponto que atinge pelo menos **90% de sensibilidade** com o menor número de falsos
   positivos, critério adequado para rastreio médico.

## Reprodutibilidade

Semente fixa (`SEED = 42`) aplicada a Python, NumPy e TensorFlow, com operações
determinísticas ativadas quando o ambiente permite. Trocar a semente, o
pré-processamento ou o mapeamento dos rótulos invalida os caches de imagens
processadas e os CSVs de split já salvos — regenere-os nesse caso.
