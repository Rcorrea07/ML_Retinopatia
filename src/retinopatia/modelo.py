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

Com `configuracao.MODELO_PARES`, a mesma rede (pesos compartilhados) vê
os dois olhos do paciente e cada olho é previsto a partir das
características dele e das do outro olho, como nas soluções de 2015 de
Jeffrey De Fauw e do time o_O. As saídas continuam por olho.

Com `configuracao.DESTILACAO`, uma terceira saída, `professor`, repete a
probabilidade de `doente` para ser comparada com a probabilidade do
professor (B4). Ela não muda a inferência.
"""

import os

import tensorflow as tf
from tensorflow.keras import applications, layers, models

from . import configuracao
from .pipeline_dados import criar_aumento_dados

BACKBONES = {
    "EfficientNetB0": applications.EfficientNetB0,
    "EfficientNetB3": applications.EfficientNetB3,
    "EfficientNetB4": applications.EfficientNetB4,
    "EfficientNetB5": applications.EfficientNetB5,
    # Leve como a B0 (cerca de 7 milhões de parâmetros) e mais precisa na
    # ImageNet. Faz o próprio pré-processamento: a entrada fica em 0–255.
    "EfficientNetV2B0": applications.EfficientNetV2B0,
    "EfficientNetV2S": applications.EfficientNetV2S
}

# Nomes das entradas do modelo de pares (dicionário do pipeline)
ENTRADA_OLHO = "olho"
ENTRADA_OUTRO_OLHO = "outro_olho"


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
        configuracao.SAIDA_GRAU: tf.keras.losses.Huber(
            delta=delta_huber
        )
    }

    pesos_perdas = {
        configuracao.SAIDA_DOENTE: 1.0,
        configuracao.SAIDA_GRAU: peso_perda_grau
    }

    # Destilação: a perda binária se divide entre o rótulo real e a
    # probabilidade do professor (entropia cruzada com alvo contínuo)
    if configuracao.DESTILACAO:
        perdas[configuracao.SAIDA_PROFESSOR] = (
            tf.keras.losses.BinaryCrossentropy()
        )
        pesos_perdas[configuracao.SAIDA_DOENTE] = configuracao.PESO_ROTULO_REAL
        pesos_perdas[configuracao.SAIDA_PROFESSOR] = (
            1.0 - configuracao.PESO_ROTULO_REAL
        )

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=taxa_aprendizado
        ),
        loss=perdas,
        loss_weights=pesos_perdas,
        metrics={
            configuracao.SAIDA_DOENTE: criar_metricas(),
            configuracao.SAIDA_GRAU: [
                tf.keras.metrics.MeanAbsoluteError(name="mae")
            ]
        },
        jit_compile=compilacao_xla
    )


def _criar_extrator(modelo_base, formato_entrada):
    """
    Augmentation (ativo só no treino), rede base e GeM: transforma uma
    imagem no vetor de características usado pelas cabeças.
    """

    entrada = layers.Input(
        shape=formato_entrada,
        name="imagem"
    )

    # Ativo só no treino; no predict a imagem passa intacta
    x = criar_aumento_dados()(entrada)

    # training=False mantém a BatchNormalization da rede base em modo
    # de inferência mesmo depois do descongelamento no fine-tuning
    x = modelo_base(x, training=False)

    caracteristicas = GeM(name="gem")(x)

    return models.Model(entrada, caracteristicas, name="extrator")


def construir_modelo(
    formato_entrada=configuracao.FORMATO_ENTRADA,
    nome_backbone=configuracao.BACKBONE,
    taxa_aprendizado=configuracao.TAXA_APRENDIZADO_FASE1,
    mostrar_resumo=True,
    pares=configuracao.MODELO_PARES
):
    """
    Monta e compila o modelo da Fase 1, com a rede base congelada.

    Com `pares=True`, o modelo recebe um dicionário com as imagens
    `olho` e `outro_olho` (o outro olho do mesmo paciente) e prevê o
    `olho`; o mesmo extrator processa as duas imagens.

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
    modelo_base = BACKBONES[nome_backbone](
        weights="imagenet",
        include_top=False,
        input_shape=formato_entrada
    )

    # Fase 1: congela a rede base e treina apenas as cabeças
    modelo_base.trainable = False

    extrator = _criar_extrator(modelo_base, formato_entrada)

    if pares:
        entradas = {
            ENTRADA_OLHO: layers.Input(
                shape=formato_entrada,
                name=ENTRADA_OLHO
            ),
            ENTRADA_OUTRO_OLHO: layers.Input(
                shape=formato_entrada,
                name=ENTRADA_OUTRO_OLHO
            )
        }

        x = layers.Concatenate(name="olho_e_outro_olho")([
            extrator(entradas[ENTRADA_OLHO]),
            extrator(entradas[ENTRADA_OUTRO_OLHO])
        ])

    else:
        entradas = layers.Input(
            shape=formato_entrada,
            name="imagem"
        )

        x = extrator(entradas)

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

    saidas = {
        configuracao.SAIDA_DOENTE: doente,
        configuracao.SAIDA_GRAU: grau
    }

    if configuracao.DESTILACAO:
        # Mesma probabilidade de `doente`, com outro alvo (o professor)
        saidas[configuracao.SAIDA_PROFESSOR] = layers.Activation(
            "linear",
            dtype="float32",
            name=configuracao.SAIDA_PROFESSOR
        )(doente)

    modelo = models.Model(
        inputs=entradas,
        outputs=saidas,
        name="retinopatia"
    )

    compilar_modelo(modelo, taxa_aprendizado)

    if mostrar_resumo:
        modelo.summary()

    print("Arquitetura montada com sucesso!")

    return modelo, modelo_base


def carregar_modelo_salvo(caminho):
    """
    Carrega um modelo .keras salvo pelo projeto (sem recompilar).
    """

    return tf.keras.models.load_model(
        caminho,
        custom_objects={"GeM": GeM},
        compile=False
    )


def construir_modelo_inferencia(
    modelo_treinado,
    formato_entrada=configuracao.FORMATO_ENTRADA,
    nome_backbone=configuracao.BACKBONE
):
    """
    Versão para o celular de um modelo treinado (de uma imagem, sem
    pares nem destilação): sem o bloco de augmentation, toda em float32
    e com uma saída só, [probabilidade de retinopatia, grau previsto].

    A rede é montada de novo em float32 (sem baixar os pesos da
    ImageNet) e recebe os pesos do modelo treinado camada a camada: a
    precisão mista e as camadas aleatórias do augmentation atrapalham a
    conversão para TFLite. A entrada continua a imagem processada em
    0–255 (`pre_processamento.processar_retina`).
    """

    politica = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy("float32")

    try:
        modelo_base = BACKBONES[nome_backbone](
            weights=None,
            include_top=False,
            input_shape=formato_entrada
        )

        entrada = layers.Input(shape=formato_entrada, name="imagem")

        x = modelo_base(entrada, training=False)
        x = GeM(name="gem")(x)

        doente = layers.Dense(1, activation="sigmoid", name=configuracao.SAIDA_DOENTE)(x)
        grau = layers.Dense(1, name=configuracao.SAIDA_GRAU)(x)

        saida = layers.Concatenate(name="doente_e_grau")([doente, grau])

        inferencia = models.Model(entrada, saida, name="retinopatia_celular")

    finally:
        tf.keras.mixed_precision.set_global_policy(politica)

    extrator = modelo_treinado.get_layer("extrator")

    # No extrator: entrada, augmentation (Sequential), rede base e GeM
    base_treinada = next(
        camada for camada in extrator.layers
        if isinstance(camada, models.Model) and camada.name != "aumento_dados"
    )

    modelo_base.set_weights(base_treinada.get_weights())
    inferencia.get_layer("gem").set_weights(extrator.get_layer("gem").get_weights())

    for nome in (configuracao.SAIDA_DOENTE, configuracao.SAIDA_GRAU):
        inferencia.get_layer(nome).set_weights(
            modelo_treinado.get_layer(nome).get_weights()
        )

    return inferencia


def salvar_modelo_final(
    modelo,
    dados,
    caminho=configuracao.CAMINHO_MODELO_FINAL
):
    """
    Salva o modelo completo (arquitetura + pesos) e confere a gravação:
    recarrega o arquivo e compara as saídas com as do modelo em memória
    num lote de `dados`. Um erro de serialização (por exemplo, da camada
    GeM ou do modelo de pares) aparece aqui, e não meses depois na API.
    """

    import numpy as np

    os.makedirs(
        os.path.dirname(caminho),
        exist_ok=True
    )

    modelo.save(caminho)

    recarregado = carregar_modelo_salvo(caminho)

    entrada = next(iter(dados))[0]

    antes = modelo(entrada, training=False)
    depois = recarregado(entrada, training=False)

    for nome in antes:
        np.testing.assert_allclose(
            np.asarray(antes[nome]),
            np.asarray(depois[nome]),
            rtol=1e-4,
            atol=1e-5
        )

    print(f"Modelo salvo e conferido (recarga idêntica): {caminho}")
