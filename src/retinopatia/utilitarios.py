"""
Utilitários gerais do projeto.

O TensorFlow é importado dentro das funções que o usam: o treino do
RETFound-Green (`retfound`, PyTorch) grava a configuração por aqui sem
carregá-lo, e ele reservaria a memória da GPU.
"""

import json
import os
import shutil
import subprocess
from datetime import datetime

from . import configuracao


def configurar_sementes(
    seed=None,
    deterministico=None
):
    """
    Configura as sementes do Python, NumPy e TensorFlow e, se
    `deterministico`, tenta ativar operações determinísticas,
    garantindo a reprodutibilidade da pesquisa.
    """

    if seed is None:
        seed = configuracao.SEED_TREINO

    if deterministico is None:
        deterministico = configuracao.OPERACOES_DETERMINISTICAS

    import tensorflow as tf

    # Mantém o valor da semente registrado no ambiente
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Configura de uma vez as sementes do Python, NumPy e TensorFlow
    tf.keras.utils.set_random_seed(seed)

    # Tenta utilizar operações determinísticas no TensorFlow
    if deterministico:
        _ativar_determinismo()

    else:
        print(
            "Operações determinísticas desligadas "
            "(OPERACOES_DETERMINISTICAS = False)."
        )

    print(
        "Bibliotecas importadas e sementes "
        "configuradas com sucesso!"
    )


def _ativar_determinismo():
    """
    Ativa as operações determinísticas, se o ambiente permitir.
    """

    import tensorflow as tf

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


def configurar_precisao_mista(ativa=None):
    """
    Ativa a precisão mista (cálculos em float16, pesos em float32).

    Precisa ser chamada antes de construir o modelo: a política vale
    para as camadas criadas depois dela. As saídas do modelo são
    criadas explicitamente em float32 para manter a perda estável.
    """

    if ativa is None:
        ativa = configuracao.PRECISAO_MISTA

    import tensorflow as tf

    politica = "mixed_float16" if ativa else "float32"

    tf.keras.mixed_precision.set_global_policy(politica)

    print(f"Política de precisão: {politica}")


def _git(*argumentos):
    """
    Executa um comando git no repositório do código e devolve a saída,
    ou None fora de um repositório.
    """

    try:
        return subprocess.run(
            ["git", *argumentos],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            check=True
        ).stdout.strip()

    except (OSError, subprocess.CalledProcessError):
        return None


def _ambiente_tensorflow():
    """
    GPU e versão do TensorFlow da execução.
    """

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")

    return {
        "gpu": [
            tf.config.experimental.get_device_details(gpu).get("device_name")
            for gpu in gpus
        ],
        "versao_tensorflow": tf.__version__
    }


def _descrever_ambiente(ambiente_ml=None):
    """
    Commit do código, alterações não commitadas e data da execução, mais
    a GPU e as bibliotecas de `ambiente_ml` (padrão: as do TensorFlow).
    """

    alteracoes = _git("status", "--porcelain", "--", ".")

    if ambiente_ml is None:
        ambiente_ml = _ambiente_tensorflow()

    return {
        "commit": _git("rev-parse", "HEAD"),
        "mensagem_commit": _git("log", "-1", "--format=%s"),
        # True quando o código foi editado no próprio Colab depois do
        # checkout: o commit sozinho não descreve o que rodou, mas a
        # cópia em codigo_src.zip sim
        "codigo_alterado_sem_commit": bool(alteracoes),
        "data_execucao": datetime.now().isoformat(timespec="seconds"),
        **ambiente_ml
    }


def salvar_configuracao(
    pasta_resultados=None,
    ambiente_ml=None
):
    """
    Registra na pasta do experimento tudo o que é preciso para saber
    como um resultado foi produzido:

    - `configuracao.json`: todas as constantes de `configuracao` (os
      nomes em maiúsculas) e o ambiente (commit, GPU, data);
    - `codigo_src.zip`: uma cópia do pacote `retinopatia` como estava na
      execução, inclusive edições feitas sem commit.

    `ambiente_ml` descreve a GPU e as bibliotecas de aprendizado (padrão:
    as do TensorFlow; o treino em PyTorch passa as suas).
    """

    if pasta_resultados is None:
        pasta_resultados = configuracao.PASTA_RESULTADOS

    os.makedirs(
        pasta_resultados,
        exist_ok=True
    )

    caminho = os.path.join(
        pasta_resultados,
        "configuracao.json"
    )

    if os.path.isfile(caminho):
        print(
            "Atenção: esta pasta já tem um experimento com o mesmo "
            "código e configuração, e ele será sobrescrito."
        )

    constantes = {
        nome: valor
        for nome, valor in vars(configuracao).items()
        if nome.isupper() and not nome.startswith("_")
    }

    constantes["AMBIENTE"] = _descrever_ambiente(ambiente_ml)

    with open(caminho, "w", encoding="utf-8") as arquivo:
        json.dump(
            constantes,
            arquivo,
            ensure_ascii=False,
            indent=2,
            # Conjuntos viram listas ordenadas; o resto vira texto
            default=lambda valor: (
                sorted(valor) if isinstance(valor, set) else str(valor)
            )
        )

    pasta_pacote = os.path.dirname(os.path.abspath(__file__))

    shutil.make_archive(
        os.path.join(pasta_resultados, "codigo_src"),
        "zip",
        root_dir=os.path.dirname(pasta_pacote),
        base_dir=os.path.basename(pasta_pacote)
    )

    print(f"Configuração e código do experimento salvos em: {pasta_resultados}")
    print(f"Commit do código: {constantes['AMBIENTE']['commit']}")

    if constantes["AMBIENTE"]["codigo_alterado_sem_commit"]:
        print(
            "Atenção: há alterações no código sem commit. Elas estão "
            "na cópia codigo_src.zip, mas não no GitHub."
        )
