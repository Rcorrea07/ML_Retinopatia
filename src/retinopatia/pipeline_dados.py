"""
Pipelines de dados com tf.data.

As imagens já vêm pré-processadas do disco (ver `pre_processamento`),
então o pipeline só lê o JPEG, embaralha o treino e monta os lotes. Os
bytes dos arquivos ficam em cache na memória depois da primeira época.

O data augmentation é definido aqui (`criar_aumento_dados`), mas roda
dentro do modelo, na GPU (ver `modelo.construir_modelo`). No tf.data
ele rodava na CPU e era o gargalo do treino: 1,2 s por lote no treino
contra 0,1 s por lote no predict, com a mesma rede.

Cada lote traz os rótulos das duas saídas do modelo:
    x, {"doente": classe binária, "grau": grau original 0 a 4}

Os conjuntos de validação e de teste nunca embaralham, para que a ordem
das predições corresponda à ordem das linhas de `df_val` e `df_teste`
na hora da avaliação.

Nenhum dos pipelines aplica rescale: a EfficientNet do Keras espera
imagens na faixa de 0 a 255 e faz a normalização internamente.
"""

import os
import time

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
    Cria o bloco de data augmentation, colocado logo depois da entrada
    do modelo. As camadas aleatórias só agem no treino (`training=True`);
    no predict, na avaliação e na TTA a imagem passa intacta.

    As camadas são criadas em float32 de propósito: com a precisão
    mista ativa, elas devolveriam float16, e a entrada da EfficientNet
    deve continuar na faixa de 0 a 255 com precisão total.
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

    return tf.keras.Sequential(
        camadas,
        name="aumento_dados"
    )


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

    Com `treino=True`, embaralha a cada época. Com `treino=False`,
    mantém a ordem do DataFrame. O data augmentation não fica aqui: ele
    roda dentro do modelo.
    """

    # Com a coluna `pasta_processada` (treino ampliado com as imagens
    # extras), cada imagem vem da sua pasta; sem ela, de `pasta_imagens`
    pastas = (
        df["pasta_processada"]
        if "pasta_processada" in df
        else [pasta_imagens] * len(df)
    )

    caminhos = [
        os.path.join(pasta, nome)
        for pasta, nome in zip(pastas, df["image"])
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
    Cria os pipelines de treino (embaralhado), validação e teste (sem
    embaralhar).

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


def medir_velocidade(dataset, passos=50):
    """
    Mede quantos lotes por segundo o pipeline entrega sozinho, sem o
    modelo. Se o pipeline levar mais por lote do que o passo de treino
    na GPU, a GPU fica esperando os dados.

    O primeiro lote é descartado da medição (inclui a montagem do
    pipeline). Na primeira época os bytes ainda estão sendo lidos do
    disco; nas seguintes vêm do cache em memória.
    """

    iterador = iter(dataset)
    next(iterador)

    inicio = time.perf_counter()

    for _ in range(passos):
        lote = next(iterador)

    duracao = time.perf_counter() - inicio

    tamanho_lote = lote[0].shape[0]

    print(
        f"\nPipeline: {duracao / passos:.3f} s por lote | "
        f"{passos * tamanho_lote / duracao:.0f} imagens por segundo"
    )
