"""
Pipelines de dados com tf.data.

As imagens já vêm pré-processadas do disco (ver `pre_processamento`),
então o pipeline só lê o JPEG, aplica data augmentation no treino e
monta os lotes. Os bytes dos arquivos ficam em cache na memória depois
da primeira época; a decodificação e o augmentation rodam em paralelo.

Cada lote traz os rótulos das duas saídas do modelo:
    x, {"doente": classe binária, "grau": grau original 0 a 4}

Os conjuntos de validação e de teste nunca embaralham, para que a ordem
das predições corresponda à ordem das linhas de `df_val` e `df_teste`
na hora da avaliação.

Nenhum dos pipelines aplica rescale: a EfficientNet do Keras espera
imagens na faixa de 0 a 255 e faz a normalização internamente.
"""

import os

import tensorflow as tf
from tensorflow.keras import layers

from . import configuracao

AUTOTUNE = tf.data.AUTOTUNE


def criar_aumento_dados(
    parametros=configuracao.AUMENTO_DADOS,
    cor_fundo=configuracao.COR_FUNDO,
    seed=configuracao.SEED
):
    """
    Cria a função de data augmentation aplicada aos lotes de treino.

    As camadas são criadas em float32 de propósito: com a precisão
    mista ativa, elas devolveriam float16, e o augmentation roda no
    pipeline de dados, não no modelo.
    """

    camadas = [
        layers.RandomFlip(
            "horizontal_and_vertical",
            seed=seed,
            dtype="float32"
        ),
        layers.RandomRotation(
            parametros["fator_rotacao"],
            fill_mode="constant",
            fill_value=cor_fundo,
            seed=seed,
            dtype="float32"
        ),
        layers.RandomZoom(
            parametros["fator_zoom"],
            fill_mode="constant",
            fill_value=cor_fundo,
            seed=seed,
            dtype="float32"
        ),
        layers.RandomBrightness(
            parametros["fator_brilho"],
            value_range=(0, 255),
            seed=seed,
            dtype="float32"
        ),
        layers.RandomContrast(
            parametros["fator_contraste"],
            seed=seed,
            dtype="float32"
        )
    ]

    def aumentar(lote):
        for camada in camadas:
            lote = camada(lote, training=True)

        return lote

    return aumentar


def _rotulos(df):
    """
    Monta o dicionário de rótulos das duas saídas, no formato (n, 1).
    """

    return {
        configuracao.SAIDA_DOENTE: (
            df["target"]
            .astype("float32")
            .values
            .reshape(-1, 1)
        ),
        configuracao.SAIDA_GRAU: (
            df["level_original"]
            .astype("float32")
            .values
            .reshape(-1, 1)
        )
    }


def criar_dataset(
    df,
    treino,
    pasta_imagens=configuracao.PASTA_IMAGENS_OTIMIZADAS,
    tamanho_imagem=configuracao.TAMANHO_IMAGEM,
    tamanho_lote=configuracao.TAMANHO_LOTE,
    seed=configuracao.SEED
):
    """
    Cria o tf.data.Dataset de um conjunto.

    Com `treino=True`, embaralha a cada época e aplica data
    augmentation. Com `treino=False`, mantém a ordem do DataFrame.
    """

    caminhos = [
        os.path.join(pasta_imagens, nome)
        for nome in df["image"]
    ]

    formato = tuple(tamanho_imagem) + (3,)

    def ler_imagem(conteudo, rotulos):
        img = tf.io.decode_jpeg(conteudo, channels=3)

        # Falha cedo se o cache tiver sido gerado com outro tamanho
        img = tf.ensure_shape(img, formato)

        return tf.cast(img, tf.float32), rotulos

    dataset = tf.data.Dataset.from_tensor_slices(
        (caminhos, _rotulos(df))
    )

    # Guarda os bytes comprimidos em memória (cerca de 1 a 2 GB), e não
    # as imagens decodificadas, que não caberiam na RAM do Colab.
    dataset = dataset.map(
        lambda caminho, rotulos: (tf.io.read_file(caminho), rotulos),
        num_parallel_calls=AUTOTUNE
    ).cache()

    if treino:
        dataset = dataset.shuffle(
            len(df),
            seed=seed,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        ler_imagem,
        num_parallel_calls=AUTOTUNE
    ).batch(tamanho_lote)

    if treino:
        aumento = criar_aumento_dados(seed=seed)

        dataset = dataset.map(
            lambda x, rotulos: (aumento(x), rotulos),
            num_parallel_calls=AUTOTUNE
        )

    return dataset.prefetch(AUTOTUNE)


def criar_geradores(
    df_treino,
    df_val,
    df_teste,
    pasta_imagens=configuracao.PASTA_IMAGENS_OTIMIZADAS,
    tamanho_imagem=configuracao.TAMANHO_IMAGEM,
    tamanho_lote=configuracao.TAMANHO_LOTE,
    seed=configuracao.SEED
):
    """
    Cria os pipelines de treino (com data augmentation), validação e
    teste (sem augmentation e sem embaralhar).

    Retorna:
        (treino_data, val_data, teste_data)
    """

    parametros = {
        "pasta_imagens": pasta_imagens,
        "tamanho_imagem": tamanho_imagem,
        "tamanho_lote": tamanho_lote,
        "seed": seed
    }

    print(f"\nTREINO: {len(df_treino)} imagens")
    treino_data = criar_dataset(df_treino, treino=True, **parametros)

    print(f"VALIDAÇÃO: {len(df_val)} imagens")
    val_data = criar_dataset(df_val, treino=False, **parametros)

    print(f"TESTE: {len(df_teste)} imagens")
    teste_data = criar_dataset(df_teste, treino=False, **parametros)

    return treino_data, val_data, teste_data


def verificar_escala(treino_data, val_data):
    """
    Confere se as imagens continuam na faixa de 0 a 255, como a
    EfficientNet espera.
    """

    x_treino = next(iter(treino_data.take(1)))[0].numpy()
    x_val = next(iter(val_data.take(1)))[0].numpy()

    print("\nVerificação da escala das imagens:")

    print(
        "Treino:",
        "formato =", x_treino.shape,
        "| tipo =", x_treino.dtype,
        "| mínimo =", x_treino.min(),
        "| máximo =", x_treino.max()
    )

    print(
        "Validação:",
        "formato =", x_val.shape,
        "| tipo =", x_val.dtype,
        "| mínimo =", x_val.min(),
        "| máximo =", x_val.max()
    )

    # A EfficientNet do Keras espera imagens aproximadamente entre 0 e 255
    assert x_treino.max() > 1.0, (
        "Erro: as imagens de treino parecem estar normalizadas entre 0 e 1."
    )

    assert x_val.max() > 1.0, (
        "Erro: as imagens de validação parecem estar normalizadas entre 0 e 1."
    )

    print("Escala das imagens verificada com sucesso!")
