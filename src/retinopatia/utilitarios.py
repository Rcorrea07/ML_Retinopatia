"""
Utilitários gerais do projeto.
"""

import os

import tensorflow as tf

from . import configuracao


def configurar_sementes(seed=configuracao.SEED):
    """
    Configura as sementes do Python, NumPy e TensorFlow e tenta ativar
    operações determinísticas, garantindo a reprodutibilidade da
    pesquisa.
    """

    # Mantém o valor da semente registrado no ambiente
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Configura de uma vez as sementes do Python, NumPy e TensorFlow
    tf.keras.utils.set_random_seed(seed)

    # Tenta utilizar operações determinísticas no TensorFlow
    try:
        tf.config.experimental.enable_op_determinism()

        print(
            "Operações determinísticas ativadas."
        )

    except Exception as erro:
        print(
            "Não foi possível ativar determinismo completo:",
            erro
        )

    print(
        "Bibliotecas importadas e sementes "
        "configuradas com sucesso!"
    )
