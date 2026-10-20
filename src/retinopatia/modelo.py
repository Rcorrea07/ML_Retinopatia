"""
Arquitetura do modelo: transfer learning com EfficientNet.

A entrada passa primeiro pelo bloco de data augmentation (ativo só no
treino, rodando na GPU). Depois, a EfficientNet pré-treinada na
ImageNet entra como extratora de características, seguida de GeM
pooling e de duas cabeças:

- `doente`: sigmoide com entropia cruzada binária. É a decisão do
  modelo (sem retinopatia x com retinopatia).
- `grau`: regressão linear do grau original de 0 a 4 com perda de
  Huber. É uma cabeça auxiliar: obriga a rede a aprender a diferença
  entre os graus, informação que a classe binária sozinha descarta.
  Todas as soluções de topo das competições do Kaggle exploraram o
  grau ordinal.

Na Fase 1 a rede base fica congelada; o descongelamento acontece no
fine-tuning (ver `treinamento.fine_tuning_fase2`).
"""

import tensorflow as tf
from tensorflow.keras import applications, layers, models

from . import configuracao
from .pipeline_dados import criar_aumento_dados

BACKBONES = {
    "EfficientNetB0": applications.EfficientNetB0,
    "EfficientNetB3": applications.EfficientNetB3,
    "EfficientNetB4": applications.EfficientNetB4,
    "EfficientNetB5": applications.EfficientNetB5,
    "EfficientNetV2S": applications.EfficientNetV2S,
    "MobileNetV3Large": applications.MobileNetV3Large
}


class GeM(layers.Layer):
    """
    Generalized Mean pooling: média das ativações elevadas a p, com p
    aprendido no treino. Usado pelo 1º lugar da APTOS 2019 no lugar da
    média global. Calcula sempre em float32 para evitar overflow da
    potência em float16.

    A classe não é registrada com `register_keras_serializable` porque o
    `%autoreload` do notebook reexecutaria o registro e o Keras recusa
    nomes duplicados. Para carregar o modelo salvo:
        tf.keras.models.load_model(caminho, custom_objects={"GeM": GeM})
    """

    def __init__(
        self,
        p_inicial=configuracao.GEM_P_INICIAL,
        eps=1e-6,
        **kwargs
    ):
        kwargs.setdefault("dtype", "float32")
        super().__init__(**kwargs)

        self.p_inicial = p_inicial
        self.eps = eps

    def build(self, input_shape):
        self.p = self.add_weight(
            name="p",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.p_inicial),
            trainable=True
        )

    def call(self, x):
        x = tf.cast(x, tf.float32)
        x = tf.pow(tf.maximum(x, self.eps), self.p)
        x = tf.reduce_mean(x, axis=[1, 2])

        return tf.pow(x, 1.0 / self.p)

    def get_config(self):
        config = super().get_config()
        config.update({
            "p_inicial": self.p_inicial,
            "eps": self.eps
        })

        return config


def criar_metricas():
    """
    Retorna novas instâncias das métricas da saída binária para cada
    compilação.
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


def compilar_modelo(
    modelo,
    taxa_aprendizado,
    peso_perda_grau=configuracao.PESO_PERDA_GRAU,
    delta_huber=configuracao.DELTA_HUBER,
    compilacao_xla=configuracao.COMPILACAO_XLA
):
    """
    Compila o modelo com as perdas e métricas das duas saídas.

    Usada na construção (Fase 1) e de novo no fine-tuning (Fase 2),
    que precisa recompilar depois de alterar `trainable`. A
    `taxa_aprendizado` pode ser um número ou um agendamento do Keras
    (a Fase 2 usa cosseno com aquecimento).
    """

    perdas = {
        configuracao.SAIDA_DOENTE: tf.keras.losses.BinaryCrossentropy(),
        configuracao.SAIDA_GRAU: tf.keras.losses.Huber(delta=delta_huber)
    }
    pesos = {configuracao.SAIDA_DOENTE: 1.0,
             configuracao.SAIDA_GRAU: peso_perda_grau}
    metricas = {
        configuracao.SAIDA_DOENTE: criar_metricas(),
        configuracao.SAIDA_GRAU: [tf.keras.metrics.MeanAbsoluteError(name="mae")]
    }
    if configuracao.MODO_GRAU == "categorico":
        perdas[configuracao.SAIDA_CLASSES] = tf.keras.losses.SparseCategoricalCrossentropy()
        pesos.update({configuracao.SAIDA_DOENTE: 0.0,
                      configuracao.SAIDA_GRAU: 0.0,
                      configuracao.SAIDA_CLASSES: configuracao.PESO_PERDA_CLASSES})
        metricas[configuracao.SAIDA_CLASSES] = [
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    if configuracao.DESTILACAO:
        perdas[configuracao.SAIDA_PROFESSOR] = tf.keras.losses.BinaryCrossentropy()
        pesos[configuracao.SAIDA_DOENTE] = configuracao.DESTILACAO["peso_bce_rotulo_real"]
        pesos[configuracao.SAIDA_PROFESSOR] = configuracao.DESTILACAO["peso_bce_professor"]
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=taxa_aprendizado
        ),
        loss=perdas,
        loss_weights=pesos,
        metrics=metricas,
        jit_compile=compilacao_xla
    )


def construir_modelo(
    formato_entrada=configuracao.FORMATO_ENTRADA,
    nome_backbone=configuracao.BACKBONE,
    taxa_aprendizado=configuracao.TAXA_APRENDIZADO_FASE1,
    mostrar_resumo=True
):
    """
    Monta e compila o modelo da Fase 1, com a rede base congelada.

    Retorna:
        (modelo, modelo_base)

    O `modelo_base` é devolvido separadamente porque o fine-tuning
    precisa dele para descongelar as camadas.
    """

    if nome_backbone not in BACKBONES:
        raise ValueError(
            f"Backbone desconhecido: {nome_backbone}. "
            f"Opções: {sorted(BACKBONES)}"
        )

    print(f"A carregar a {nome_backbone} e a construir o modelo...")

    # Carrega a rede base sem a camada classificadora original
    extras = ({"alpha": configuracao.ALPHA_MOBILENET,
               "include_preprocessing": True}
              if nome_backbone == "MobileNetV3Large" else {})
    modelo_base = BACKBONES[nome_backbone](
        weights=configuracao.PESOS_INICIAIS,
        include_top=False,
        input_shape=formato_entrada,
        **extras
    )

    # Fase 1: congela a rede base e treina apenas as cabeças
    modelo_base.trainable = False

    entrada = layers.Input(
        shape=formato_entrada,
        name="imagem"
    )

    # Ativo só no treino; no predict a imagem passa intacta
    x = criar_aumento_dados()(entrada)

    # training=False mantém a BatchNormalization da rede base em modo
    # de inferência mesmo depois do descongelamento no fine-tuning
    x = modelo_base(x, training=False)

    x = (GeM(name="gem")(x) if configuracao.POOLING == "GeM"
         else layers.GlobalAveragePooling2D(name="pooling_medio", dtype="float32")(x))

    x = layers.Dropout(
        configuracao.TAXA_DROPOUT,
        name="dropout"
    )(x)

    # As saídas ficam em float32 mesmo com precisão mista, para manter
    # o cálculo da perda numericamente estável
    doente = layers.Dense(
        1,
        activation="sigmoid",
        dtype="float32",
        name=configuracao.SAIDA_DOENTE
    )(x)

    grau = layers.Dense(
        1,
        dtype="float32",
        name=configuracao.SAIDA_GRAU
    )(x)

    saidas = {configuracao.SAIDA_DOENTE: doente, configuracao.SAIDA_GRAU: grau}
    if configuracao.DESTILACAO:
        saidas[configuracao.SAIDA_PROFESSOR] = layers.Activation(
            "linear", dtype="float32", name=configuracao.SAIDA_PROFESSOR)(doente)
    if configuracao.MODO_GRAU == "categorico":
        classes = layers.Dense(configuracao.NUMERO_GRAUS, activation="softmax",
                               dtype="float32", name=configuracao.SAIDA_CLASSES)(x)
        # Camadas fixas evitam Lambda e mantêm os nomes das saídas antigas.
        saidas = {
            configuracao.SAIDA_CLASSES: classes,
            configuracao.SAIDA_DOENTE: layers.Dense(
                1, use_bias=False, trainable=False, dtype="float32",
                kernel_initializer=tf.keras.initializers.Constant([[0], [1], [1], [1], [1]]),
                name=configuracao.SAIDA_DOENTE)(classes),
            configuracao.SAIDA_GRAU: layers.Dense(
                1, use_bias=False, trainable=False, dtype="float32",
                kernel_initializer=tf.keras.initializers.Constant([[0], [1], [2], [3], [4]]),
                name=configuracao.SAIDA_GRAU)(classes)
        }
    modelo = models.Model(
        inputs=entrada,
        outputs=saidas,
        name="retinopatia"
    )

    compilar_modelo(modelo, taxa_aprendizado)

    if mostrar_resumo:
        modelo.summary()

    print("Arquitetura montada com sucesso!")

    return modelo, modelo_base
