"""
Pipelines de dados do Keras.

O gerador de treino aplica data augmentation; o de validação é limpo e
nunca embaralha, para que a ordem das imagens corresponda a
`val_data.classes` na hora de montar a matriz de confusão.

Nenhum dos geradores aplica rescale: a EfficientNet do Keras espera
imagens na faixa de 0 a 255 e faz a normalização internamente.
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from . import configuracao


def criar_geradores(
    df_treino,
    df_val,
    pasta_imagens=configuracao.PASTA_IMAGENS_OTIMIZADAS,
    tamanho_imagem=configuracao.TAMANHO_IMAGEM,
    tamanho_lote=configuracao.TAMANHO_LOTE,
    seed=configuracao.SEED
):
    """
    Cria os geradores de treino (com data augmentation) e de validação
    (sem augmentation).

    Retorna:
        (treino_data, val_data)
    """

    # Gerador de TREINO (com data augmentation)
    gerador_treino = ImageDataGenerator(
        **configuracao.AUMENTO_DADOS
    )

    # Gerador de VALIDAÇÃO (limpo, sem augmentation e sem rescale)
    gerador_val = ImageDataGenerator()

    print("\nA preparar imagens de TREINO...")

    treino_data = gerador_treino.flow_from_dataframe(
        dataframe=df_treino,
        directory=pasta_imagens,
        x_col="image",
        y_col="target",
        class_mode="binary",
        target_size=tamanho_imagem,
        batch_size=tamanho_lote,
        seed=seed,

        # Fundamental embaralhar no treino
        shuffle=True
    )

    print("\nA preparar imagens de VALIDAÇÃO...")

    val_data = gerador_val.flow_from_dataframe(
        dataframe=df_val,
        directory=pasta_imagens,
        x_col="image",
        y_col="target",
        class_mode="binary",
        target_size=tamanho_imagem,
        batch_size=tamanho_lote,
        seed=seed,

        # Nunca embaralhar a validação, para podermos criar a matriz
        # de confusão correta depois
        shuffle=False
    )

    return treino_data, val_data


def verificar_escala(treino_data, val_data):
    """
    Confere se as imagens continuam na faixa de 0 a 255, como a
    EfficientNet espera, e devolve os geradores ao início.
    """

    x_treino, y_treino = next(treino_data)
    x_val, y_val = next(val_data)

    print("\nVerificação da escala das imagens:")

    print(
        "Treino:",
        "tipo =", x_treino.dtype,
        "| mínimo =", x_treino.min(),
        "| máximo =", x_treino.max()
    )

    print(
        "Validação:",
        "tipo =", x_val.dtype,
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

    # Retorna os geradores ao início antes do treinamento
    treino_data.reset()
    val_data.reset()

    print("Escala das imagens verificada com sucesso!")
