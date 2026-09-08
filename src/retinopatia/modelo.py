"""
Arquitetura do modelo: transfer learning com EfficientNetB0.

A EfficientNetB0 pré-treinada na ImageNet entra como extratora de
características, seguida de uma cabeça densa binária. Na Fase 1 a rede
base fica congelada; o descongelamento acontece no fine-tuning
(ver `treinamento.fine_tuning_fase2`).
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

from . import configuracao


def criar_metricas():
    """
    Retorna novas instâncias das métricas para cada compilação.
    """
    return [
        tf.keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),
        tf.keras.metrics.Precision(
            name="precision"
        ),
        tf.keras.metrics.Recall(
            name="sensitivity"
        ),
        tf.keras.metrics.AUC(
            name="roc_auc",
            curve="ROC"
        ),
        tf.keras.metrics.AUC(
            name="pr_auc",
            curve="PR"
        )
    ]


def construir_modelo(
    formato_entrada=configuracao.FORMATO_ENTRADA,
    taxa_aprendizado=configuracao.TAXA_APRENDIZADO_FASE1,
    mostrar_resumo=True
):
    """
    Monta e compila o modelo da Fase 1, com a EfficientNetB0 congelada.

    Retorna:
        (modelo, modelo_base)

    O `modelo_base` é devolvido separadamente porque o fine-tuning
    precisa dele para descongelar as camadas.
    """

    print("A carregar a EfficientNetB0 e a construir o modelo...")

    # Carrega a EfficientNetB0 sem a camada classificadora original
    modelo_base = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=formato_entrada
    )

    # Fase 1: congela a EfficientNet e treina apenas a cabeça
    modelo_base.trainable = False

    modelo = models.Sequential([
        modelo_base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(
            configuracao.UNIDADES_DENSA,
            activation="relu"
        ),
        layers.Dropout(configuracao.TAXA_DROPOUT),
        layers.Dense(
            1,
            activation="sigmoid"
        )
    ])

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=taxa_aprendizado
        ),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=criar_metricas()
    )

    if mostrar_resumo:
        modelo.summary()

    print("Arquitetura montada com sucesso!")

    return modelo, modelo_base
