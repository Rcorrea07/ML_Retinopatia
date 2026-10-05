"""
Treino do modelo em duas fases.

Fase 1: a EfficientNet fica congelada e apenas as cabeças são treinadas,
com uma taxa de aprendizado alta.

Fase 2 (fine-tuning): a rede base é descongelada, mas as camadas de
BatchNormalization continuam congeladas — requisito conhecido ao fazer
fine-tuning de EfficientNet — e o treino segue com uma taxa de
aprendizado bem menor.

As duas fases usam callbacks com a mesma estrutura, criada por
`criar_callbacks`, mudando apenas o arquivo de pesos e as paciências.

O desbalanceamento entre as classes é compensado com pesos por amostra
(`sample_weight`) acrescentados ao pipeline de treino: o argumento
`class_weight` do Keras não funciona em modelos com mais de uma saída.
"""

import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from . import configuracao
from .modelo import compilar_modelo


def calcular_pesos_classe(df_treino):
    """
    Calcula os pesos das classes de forma balanceada, para compensar o
    desbalanceamento entre imagens saudáveis e doentes.

    Retorna um dicionário no formato {0: peso, 1: peso}.
    """

    print("A calcular os pesos matemáticos das classes...")

    # Converte a coluna target para números inteiros
    classes_reais_treino = (
        df_treino["target"]
        .astype(int)
        .values
    )

    # Descobre quais classes realmente existem no treino
    # Neste projeto, o esperado é: [0, 1]
    classes = np.unique(classes_reais_treino)

    # Verifica se as duas classes estão presentes
    if set(classes) != {0, 1}:
        raise ValueError(
            f"Classes inesperadas no treino: {classes}. "
            "O esperado era encontrar as classes 0 e 1."
        )

    # Calcula os pesos de acordo com a quantidade
    # de exemplos de cada classe
    pesos = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=classes_reais_treino
    )

    # Associa cada classe diretamente ao seu peso
    # Exemplo: {0: 0.65, 1: 2.10}
    pesos_dicionario = dict(
        zip(
            classes.tolist(),
            pesos.tolist()
        )
    )

    print("\nPesos calculados:")

    for classe, peso in pesos_dicionario.items():
        print(
            f"Classe {classe} "
            f"({configuracao.NOMES_CLASSES[classe]}): "
            f"peso {peso:.4f}"
        )

    print(
        "\nPesos calculados com sucesso! "
        "A classe menos frequente receberá maior peso durante o treinamento."
    )

    return pesos_dicionario


def adicionar_pesos_amostra(treino_data, pesos_dicionario):
    """
    Acrescenta a cada lote de treino o peso de cada amostra, de acordo
    com a sua classe binária, no formato (x, rótulos, pesos).

    O mesmo peso vale para as duas saídas do modelo.
    """

    pesos = tf.constant(
        [pesos_dicionario[0], pesos_dicionario[1]],
        dtype=tf.float32
    )

    def com_pesos(x, rotulos):
        classes = tf.cast(
            tf.reshape(rotulos[configuracao.SAIDA_DOENTE], [-1]),
            tf.int32
        )

        return x, rotulos, tf.gather(pesos, classes)

    return treino_data.map(
        com_pesos,
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)


def criar_callbacks(
    caminho_pesos,
    paciencia_early_stop,
    paciencia_reduce_lr,
    lr_minimo,
    metrica=configuracao.METRICA_MONITORADA,
    fator_reduce_lr=configuracao.FATOR_REDUCE_LR
):
    """
    Cria os callbacks de monitorização do treino.

    O checkpoint e o early stopping acompanham a métrica de validação
    escolhida (PR-AUC por padrão); a redução de learning rate acompanha
    o erro de validação.
    """

    checkpoint = ModelCheckpoint(
        caminho_pesos,
        monitor=metrica,
        mode="max",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor=metrica,
        mode="max",
        patience=paciencia_early_stop,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=fator_reduce_lr,
        patience=paciencia_reduce_lr,
        min_lr=lr_minimo,
        verbose=1
    )

    return [
        checkpoint,
        early_stop,
        reduce_lr
    ]


def criar_callbacks_fase1():
    """
    Callbacks da Fase 1, com os valores definidos em `configuracao`.
    """

    return criar_callbacks(
        caminho_pesos=configuracao.CAMINHO_PESOS_FASE1,
        paciencia_early_stop=configuracao.PACIENCIA_EARLY_STOP_FASE1,
        paciencia_reduce_lr=configuracao.PACIENCIA_REDUCE_LR_FASE1,
        lr_minimo=configuracao.LR_MINIMO_FASE1
    )


def criar_callbacks_fase2():
    """
    Callbacks da Fase 2, com paciências maiores e learning rate mínimo
    menor que os da Fase 1.
    """

    return criar_callbacks(
        caminho_pesos=configuracao.CAMINHO_PESOS_FASE2,
        paciencia_early_stop=configuracao.PACIENCIA_EARLY_STOP_FASE2,
        paciencia_reduce_lr=configuracao.PACIENCIA_REDUCE_LR_FASE2,
        lr_minimo=configuracao.LR_MINIMO_FASE2
    )


def treinar_fase1(
    modelo,
    treino_data,
    val_data,
    pesos_dicionario,
    callbacks,
    epocas=configuracao.EPOCAS_FASE1,
    caminho_pesos=configuracao.CAMINHO_PESOS_FASE1
):
    """
    Treina apenas as cabeças da rede e recarrega o melhor checkpoint ao
    final, deixando o modelo pronto para a avaliação.

    Retorna o histórico do treino.
    """

    print("Iniciando o treinamento...")

    historico = modelo.fit(
        # Os pesos por amostra compensam o desbalanceamento das classes
        adicionar_pesos_amostra(treino_data, pesos_dicionario),
        validation_data=val_data,
        epochs=epocas,
        callbacks=callbacks
    )

    print("\n Treino da Fase 1 concluído com sucesso!")

    print("Carregando o melhor checkpoint da Fase 1 para avaliação...")

    modelo.load_weights(caminho_pesos)

    return historico


def fine_tuning_fase2(
    modelo,
    modelo_base,
    treino_data,
    val_data,
    pesos_dicionario,
    callbacks,
    epocas=configuracao.EPOCAS_FASE2,
    taxa_aprendizado=configuracao.TAXA_APRENDIZADO_FASE2,
    caminho_pesos_fase1=configuracao.CAMINHO_PESOS_FASE1,
    caminho_pesos_fase2=configuracao.CAMINHO_PESOS_FASE2
):
    """
    Executa o fine-tuning: recupera os pesos da Fase 1, descongela a
    rede base mantendo a BatchNormalization congelada, recompila com uma
    taxa de aprendizado menor e treina novamente.

    Retorna o histórico do treino da Fase 2.
    """

    print("Recuperando os melhores pesos da Fase 1...")

    modelo.load_weights(caminho_pesos_fase1)

    print("Descongelando a rede e mantendo BatchNormalization congelada...")

    # Descongela o modelo base
    modelo_base.trainable = True

    # Mantém as camadas BatchNormalization congeladas
    for layer in modelo_base.layers:
        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):
            layer.trainable = False

    # Recompila obrigatoriamente depois de alterar trainable
    compilar_modelo(modelo, taxa_aprendizado)

    print("Iniciando o treinamento da Fase 2...")

    historico_fase2 = modelo.fit(
        adicionar_pesos_amostra(treino_data, pesos_dicionario),
        validation_data=val_data,
        epochs=epocas,
        callbacks=callbacks
    )

    print("Carregando o melhor checkpoint da Fase 2 para avaliação...")

    modelo.load_weights(caminho_pesos_fase2)

    return historico_fase2
