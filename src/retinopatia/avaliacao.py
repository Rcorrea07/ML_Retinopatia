"""
Avaliação do modelo: curvas de treino, escolha do limiar de decisão,
matriz de confusão e relatório de classificação.

O limiar não é fixado em 0.5. Ele é escolhido na curva ROC como o ponto
que atinge pelo menos a sensibilidade desejada com o menor número de
falsos positivos — a escolha adequada para rastreio médico, onde deixar
de detectar um doente custa mais caro do que um alarme falso.

No notebook original este bloco aparecia duplicado (uma vez após a
Fase 1 e outra após a Fase 2). Aqui existe uma única função,
`avaliar_modelo`, usada nas duas fases: o parâmetro `nome_fase` apenas
diferencia os títulos e as mensagens.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve
)

from . import configuracao


def plotar_historico(historico, nome_fase):
    """
    Desenha a evolução da acurácia e do erro ao longo das épocas.
    """

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)

    plt.plot(
        historico.history["accuracy"],
        label="Treino",
        linewidth=2
    )

    plt.plot(
        historico.history["val_accuracy"],
        label="Validação",
        linewidth=2
    )

    plt.title(f"Evolução da Acurácia — {nome_fase}")
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia")
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
    nome_fase
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
        f"Matriz de Confusão — {nome_fase}\n"
        f"Limiar = {limiar:.4f}"
    )

    plt.xlabel("Classe predita")
    plt.ylabel("Classe verdadeira")
    plt.tight_layout()
    plt.show()

    return matriz


def avaliar_modelo(
    modelo,
    val_data,
    historico,
    nome_fase,
    sensibilidade_desejada=configuracao.SENSIBILIDADE_DESEJADA
):
    """
    Executa a avaliação completa de uma fase de treino: gráficos do
    histórico, escolha do limiar na validação, matriz de confusão e
    relatório de classificação.

    Retorna um dicionário com o limiar escolhido, a sensibilidade e a
    especificidade estimadas e as predições da validação.
    """

    # 1. GRÁFICOS DO TREINAMENTO
    plotar_historico(historico, nome_fase)

    # 2. ESCOLHA DO LIMIAR NA VALIDAÇÃO
    print("\nGerando probabilidades na validação...")

    # Reseta o gerador para garantir a ordem exata das imagens
    val_data.reset()

    predicoes = modelo.predict(
        val_data,
        verbose=1
    ).ravel()

    classes_verdadeiras = val_data.classes

    if len(np.unique(classes_verdadeiras)) != 2:
        raise ValueError(
            "A validação precisa possuir as classes 0 e 1."
        )

    limiar, sensibilidade_obtida, especificidade_obtida = escolher_limiar(
        classes_verdadeiras,
        predicoes,
        sensibilidade_desejada
    )

    print(f"\nLimiar escolhido na {nome_fase}: {limiar:.4f}")

    if sensibilidade_obtida is not None:
        print(
            f"Sensibilidade estimada: "
            f"{sensibilidade_obtida:.4f}"
        )

        print(
            f"Especificidade estimada: "
            f"{especificidade_obtida:.4f}"
        )

    else:
        print(
            "Nenhum limiar atingiu a sensibilidade desejada. "
            "Foi utilizado o limiar padrão de 0.5."
        )

    classes_preditas = (
        predicoes >= limiar
    ).astype(int)

    # 3. MATRIZ DE CONFUSÃO
    matriz = plotar_matriz_confusao(
        classes_verdadeiras,
        classes_preditas,
        limiar,
        nome_fase
    )

    # 4. RELATÓRIO
    print(f"\nRelatório de classificação da {nome_fase}:")

    print(
        classification_report(
            classes_verdadeiras,
            classes_preditas,
            target_names=configuracao.ROTULOS_RELATORIO,
            zero_division=0
        )
    )

    return {
        "limiar": limiar,
        "sensibilidade": sensibilidade_obtida,
        "especificidade": especificidade_obtida,
        "predicoes": predicoes,
        "classes_preditas": classes_preditas,
        "matriz_confusao": matriz
    }
