"""
Avaliação do modelo: curvas de treino, escolha do limiar de decisão,
matriz de confusão e relatório de classificação, por olho e por
paciente.

O limiar não é fixado em 0.5. Ele é escolhido na curva ROC como o ponto
que atinge pelo menos a sensibilidade desejada com o menor número de
falsos positivos — a escolha adequada para rastreio médico, onde deixar
de detectar um doente custa mais caro do que um alarme falso.

O limiar é escolhido na VALIDAÇÃO e aplicado ao TESTE. Os números da
validação são otimistas (o limiar foi ajustado neles); os do teste são
a estimativa honesta do desempenho.

Além da avaliação por olho, há a avaliação por paciente: a
probabilidade do paciente é a maior entre as dos seus dois olhos, como
na regra clínica de encaminhar o paciente se qualquer olho tiver
retinopatia. As melhores soluções da competição de 2015 ganharam
desempenho combinando os dois olhos.

Uma única função, `avaliar_modelo`, serve às duas fases de treino: o
parâmetro `nome_fase` apenas diferencia os títulos e as mensagens.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from . import configuracao


def _serie_historico(historico, nomes):
    """
    Devolve a primeira série do histórico encontrada entre os nomes
    possíveis (os nomes das métricas variam entre versões do Keras).
    """

    for nome in nomes:
        if nome in historico.history:
            return historico.history[nome]

    raise KeyError(
        f"Nenhuma das métricas {nomes} está no histórico. "
        f"Disponíveis: {sorted(historico.history)}"
    )


def plotar_historico(historico, nome_fase):
    """
    Desenha a evolução da PR-AUC da saída binária e do erro total ao
    longo das épocas.
    """

    saida = configuracao.SAIDA_DOENTE

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)

    plt.plot(
        _serie_historico(historico, [f"{saida}_pr_auc", "pr_auc"]),
        label="Treino",
        linewidth=2
    )

    plt.plot(
        _serie_historico(historico, [f"val_{saida}_pr_auc", "val_pr_auc"]),
        label="Validação",
        linewidth=2
    )

    plt.title(f"Evolução da PR-AUC — {nome_fase}")
    plt.xlabel("Épocas")
    plt.ylabel("PR-AUC")
    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(
        historico.history["loss"],
        label="Treino",
        linewidth=2
    )

    plt.plot(
        historico.history["val_loss"],
        label="Validação",
        linewidth=2
    )

    plt.title(f"Evolução do Erro — {nome_fase}")
    plt.xlabel("Épocas")
    plt.ylabel("Erro")
    plt.legend()

    plt.tight_layout()
    plt.show()


def _variantes_tta(tta):
    """
    Transformações aplicadas às imagens na test-time augmentation.
    """

    identidade = [lambda x: x]

    if not tta:
        return identidade

    return identidade + [
        tf.image.flip_left_right,
        tf.image.flip_up_down,
        lambda x: tf.image.flip_up_down(tf.image.flip_left_right(x))
    ]


def prever_probabilidades(
    modelo,
    dados,
    tta=configuracao.TTA_ATIVO
):
    """
    Gera a probabilidade de retinopatia de cada imagem, na ordem do
    pipeline (que não embaralha). Com TTA, devolve a média das
    predições da imagem original e de seus espelhamentos.
    """

    predicoes = []

    for transformar in _variantes_tta(tta):
        # O pipeline devolve (imagens, rótulos); só as imagens interessam
        imagens = dados.map(
            lambda x, *_, f=transformar: f(x)
        )

        saida = modelo.predict(imagens, verbose=1)

        if isinstance(saida, dict):
            saida = saida[configuracao.SAIDA_DOENTE]

        predicoes.append(np.asarray(saida).ravel())

    return np.mean(predicoes, axis=0)


def agregar_por_paciente(df, probabilidades):
    """
    Combina os dois olhos de cada paciente: a classe do paciente é 1 se
    qualquer olho estiver doente, e a probabilidade é a maior entre as
    dos dois olhos.

    Retorna:
        (classes_paciente, probabilidades_paciente)
    """

    por_olho = pd.DataFrame({
        "patient_id": df["patient_id"].values,
        "target": df["target"].astype(int).values,
        "probabilidade": probabilidades
    })

    por_paciente = por_olho.groupby("patient_id").max()

    return (
        por_paciente["target"].values,
        por_paciente["probabilidade"].values
    )


def escolher_limiar(
    classes_verdadeiras,
    predicoes,
    sensibilidade_desejada=configuracao.SENSIBILIDADE_DESEJADA
):
    """
    Escolhe, na curva ROC, o limiar que atinge pelo menos a
    sensibilidade desejada com a menor taxa de falsos positivos.

    Retorna:
        (limiar, sensibilidade_obtida, especificidade_obtida)

    Se nenhum limiar atingir a sensibilidade desejada, devolve o limiar
    padrão de 0.5 e None nas outras duas posições.
    """

    fpr, tpr, thresholds = roc_curve(
        classes_verdadeiras,
        predicoes
    )

    indices_validos = np.where(
        tpr >= sensibilidade_desejada
    )[0]

    if len(indices_validos) > 0:
        melhor_indice = indices_validos[
            np.argmin(fpr[indices_validos])
        ]

        return (
            thresholds[melhor_indice],
            tpr[melhor_indice],
            1 - fpr[melhor_indice]
        )

    return 0.5, None, None


def plotar_matriz_confusao(
    classes_verdadeiras,
    classes_preditas,
    limiar,
    titulo
):
    """
    Desenha a matriz de confusão no limiar escolhido.
    """

    matriz = confusion_matrix(
        classes_verdadeiras,
        classes_preditas,
        labels=[0, 1]
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        matriz,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=configuracao.ROTULOS_MATRIZ,
        yticklabels=configuracao.ROTULOS_MATRIZ
    )

    plt.title(
        f"Matriz de Confusão — {titulo}\n"
        f"Limiar = {limiar:.4f}"
    )

    plt.xlabel("Classe predita")
    plt.ylabel("Classe verdadeira")
    plt.tight_layout()
    plt.show()

    return matriz


def relatar_desempenho(
    classes_verdadeiras,
    probabilidades,
    limiar,
    titulo
):
    """
    Mostra ROC-AUC, PR-AUC, sensibilidade e especificidade no limiar,
    a matriz de confusão e o relatório de classificação.

    Retorna um dicionário com as métricas.
    """

    if len(np.unique(classes_verdadeiras)) != 2:
        raise ValueError(
            f"{titulo}: o conjunto precisa possuir as classes 0 e 1."
        )

    classes_preditas = (
        probabilidades >= limiar
    ).astype(int)

    matriz = plotar_matriz_confusao(
        classes_verdadeiras,
        classes_preditas,
        limiar,
        titulo
    )

    vn, fp, fn, vp = matriz.ravel()

    metricas = {
        "roc_auc": roc_auc_score(classes_verdadeiras, probabilidades),
        "pr_auc": average_precision_score(
            classes_verdadeiras,
            probabilidades
        ),
        "sensibilidade": vp / (vp + fn),
        "especificidade": vn / (vn + fp),
        "limiar": limiar,
        "matriz_confusao": matriz
    }

    print(f"\n{titulo}:")
    print(f"ROC-AUC: {metricas['roc_auc']:.4f}")
    print(f"PR-AUC: {metricas['pr_auc']:.4f}")
    print(
        f"Sensibilidade no limiar {limiar:.4f}: "
        f"{metricas['sensibilidade']:.4f}"
    )
    print(
        f"Especificidade no limiar {limiar:.4f}: "
        f"{metricas['especificidade']:.4f}"
    )

    print(
        classification_report(
            classes_verdadeiras,
            classes_preditas,
            target_names=configuracao.ROTULOS_RELATORIO,
            zero_division=0
        )
    )

    return metricas


def _escolher_e_informar_limiar(
    classes_verdadeiras,
    probabilidades,
    nivel,
    sensibilidade_desejada
):
    """
    Escolhe o limiar de um nível (olho ou paciente) e informa o
    resultado.
    """

    limiar, sensibilidade, especificidade = escolher_limiar(
        classes_verdadeiras,
        probabilidades,
        sensibilidade_desejada
    )

    print(f"\nLimiar escolhido por {nivel}: {limiar:.4f}")

    if sensibilidade is None:
        print(
            "Nenhum limiar atingiu a sensibilidade desejada. "
            "Foi utilizado o limiar padrão de 0.5."
        )

    return limiar


def avaliar_modelo(
    modelo,
    val_data,
    df_val,
    historico,
    nome_fase,
    sensibilidade_desejada=configuracao.SENSIBILIDADE_DESEJADA
):
    """
    Executa a avaliação completa de uma fase de treino na validação:
    gráficos do histórico, escolha dos limiares (por olho e por
    paciente), matrizes de confusão e relatórios.

    Retorna um dicionário com os limiares escolhidos, as métricas e as
    predições da validação. Os limiares são reutilizados por
    `avaliar_no_teste`.
    """

    # 1. GRÁFICOS DO TREINAMENTO
    plotar_historico(historico, nome_fase)

    # 2. PROBABILIDADES NA VALIDAÇÃO
    print("\nGerando probabilidades na validação...")

    predicoes = prever_probabilidades(modelo, val_data)

    classes_olho = df_val["target"].astype(int).values

    classes_paciente, predicoes_paciente = agregar_por_paciente(
        df_val,
        predicoes
    )

    # 3. ESCOLHA DOS LIMIARES
    limiar_olho = _escolher_e_informar_limiar(
        classes_olho,
        predicoes,
        "olho",
        sensibilidade_desejada
    )

    limiar_paciente = _escolher_e_informar_limiar(
        classes_paciente,
        predicoes_paciente,
        "paciente",
        sensibilidade_desejada
    )

    # 4. MATRIZES E RELATÓRIOS
    print(
        "\nAtenção: os limiares foram escolhidos nesta validação, "
        "então estes números são otimistas. Use o teste para a "
        "estimativa final."
    )

    metricas_olho = relatar_desempenho(
        classes_olho,
        predicoes,
        limiar_olho,
        f"{nome_fase} — validação, por olho"
    )

    metricas_paciente = relatar_desempenho(
        classes_paciente,
        predicoes_paciente,
        limiar_paciente,
        f"{nome_fase} — validação, por paciente"
    )

    return {
        "limiar_olho": limiar_olho,
        "limiar_paciente": limiar_paciente,
        "metricas_olho": metricas_olho,
        "metricas_paciente": metricas_paciente,
        "predicoes": predicoes
    }


def avaliar_no_teste(
    modelo,
    teste_data,
    df_teste,
    resultado_validacao,
    nome="Teste"
):
    """
    Avalia o modelo no conjunto de teste usando os limiares escolhidos
    na validação (o resultado de `avaliar_modelo`). É a estimativa
    honesta do desempenho do modelo.

    Retorna um dicionário com as métricas por olho e por paciente e as
    predições do teste.
    """

    print("\nGerando probabilidades no teste...")

    predicoes = prever_probabilidades(modelo, teste_data)

    classes_paciente, predicoes_paciente = agregar_por_paciente(
        df_teste,
        predicoes
    )

    metricas_olho = relatar_desempenho(
        df_teste["target"].astype(int).values,
        predicoes,
        resultado_validacao["limiar_olho"],
        f"{nome} — por olho"
    )

    metricas_paciente = relatar_desempenho(
        classes_paciente,
        predicoes_paciente,
        resultado_validacao["limiar_paciente"],
        f"{nome} — por paciente"
    )

    return {
        "metricas_olho": metricas_olho,
        "metricas_paciente": metricas_paciente,
        "predicoes": predicoes
    }
