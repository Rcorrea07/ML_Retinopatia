"""
Regras de medida compartilhadas pela avaliação, pelo diagnóstico e pela
análise das predições salvas.

Só numpy, pandas e scikit-learn: nada de TensorFlow, gráficos, `print`
ou arquivos. Cada regra existe uma vez aqui e os outros módulos a usam:

- o ponto de operação: o limiar de menor taxa de falsos positivos entre
  os que dão a sensibilidade desejada (`ponto_na_sensibilidade`);
- a decisão por paciente: doente se algum olho for, com a maior
  probabilidade dos dois olhos (`por_paciente`);
- o kappa quadrático da cabeça de grau (`kappa_grau`);
- as métricas num limiar fixo e os intervalos de confiança por
  bootstrap de pacientes (`metricas_no_limiar`, `intervalos_bootstrap`).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    roc_auc_score,
    roc_curve
)

# Formas de combinar os dois olhos de um paciente
AGREGACOES = ("max", "media", "max_media")


def ponto_na_sensibilidade(classes, probabilidades, sensibilidade_desejada):
    """
    Na curva ROC, o ponto que atinge pelo menos `sensibilidade_desejada`
    com a menor taxa de falsos positivos.

    Retorna:
        (limiar, sensibilidade, especificidade), ou (None, None, None)
        quando nenhum limiar atinge a sensibilidade
    """

    fpr, tpr, limiares = roc_curve(classes, probabilidades)

    validos = np.where(tpr >= sensibilidade_desejada)[0]

    if len(validos) == 0:
        return None, None, None

    melhor = validos[np.argmin(fpr[validos])]

    return limiares[melhor], tpr[melhor], 1 - fpr[melhor]


def agregar_olhos(valores, funcao="max"):
    """
    Combina os valores dos olhos de um paciente: "max" (o maior),
    "media" ou "max_media" (média dos dois anteriores).
    """

    if funcao == "max":
        return valores.max()

    if funcao == "media":
        return valores.mean()

    if funcao == "max_media":
        return (valores.max() + valores.mean()) / 2

    raise ValueError(f"Agregação desconhecida: {funcao}")


def por_paciente(pacientes, classes, valores, funcao="max"):
    """
    Uma linha por paciente, na ordem dos identificadores: a classe é 1
    se algum olho for doente, e o valor combina os dos olhos
    (`agregar_olhos`).

    Retorna:
        DataFrame indexado por `patient_id`, com as colunas `target` e
        `valor`
    """

    olhos = pd.DataFrame({
        "patient_id": np.asarray(pacientes),
        "target": np.asarray(classes).astype(int),
        "valor": np.asarray(valores)
    })

    agrupado = olhos.groupby("patient_id")

    if funcao == "max":
        valor = agrupado["valor"].max()
    else:
        valor = agrupado["valor"].agg(lambda parte: agregar_olhos(parte, funcao))

    return pd.DataFrame({
        "target": agrupado["target"].max(),
        "valor": valor
    })


def kappa_grau(niveis_reais, graus_previstos):
    """
    Kappa quadrático da cabeça de grau: a regressão arredondada para o
    inteiro mais próximo e limitada a 0–4. É a métrica do Kaggle EyePACS
    2015 (1º lugar: 0,85 no teste da competição).
    """

    previstos = np.clip(np.rint(graus_previstos), 0, 4).astype(int)

    return cohen_kappa_score(
        np.asarray(niveis_reais).astype(int),
        previstos,
        weights="quadratic"
    )


def metricas_no_limiar(classes, probabilidades, limiar):
    """
    ROC-AUC, PR-AUC, sensibilidade e especificidade no `limiar`.
    """

    classes = np.asarray(classes)
    preditas = np.asarray(probabilidades) >= limiar

    return {
        "roc_auc": roc_auc_score(classes, probabilidades),
        "pr_auc": average_precision_score(classes, probabilidades),
        "sensibilidade": preditas[classes == 1].mean(),
        "especificidade": 1 - preditas[classes == 0].mean()
    }


def intervalos_bootstrap(
    tabela,
    limiar_olho,
    limiar_paciente,
    numero_reamostragens,
    seed,
    coluna="prob_doente"
):
    """
    Intervalos de confiança de 95% das métricas por olho e por paciente,
    reamostrando pacientes com reposição: os dois olhos de um paciente
    entram ou saem juntos, porque não são independentes. Os limiares são
    os escolhidos na validação.

    Retorna:
        DataFrame indexado por (nível, métrica), com as colunas
        `estimativa`, `ic95_inferior` e `ic95_superior`
    """

    pacientes = tabela.groupby("patient_id").agg(
        target=("target", "max"),
        probabilidade=(coluna, "max")
    )

    linhas_por_paciente = tabela.groupby("patient_id").indices

    indices_olhos = [
        linhas_por_paciente[paciente]
        for paciente in pacientes.index
    ]

    classes_olho = tabela["target"].astype(int).values
    probs_olho = tabela[coluna].values
    classes_paciente = pacientes["target"].astype(int).values
    probs_paciente = pacientes["probabilidade"].values

    gerador = np.random.default_rng(seed)
    numero_pacientes = len(pacientes)

    amostras = {"olho": [], "paciente": []}

    for _ in range(numero_reamostragens):
        sorteados = gerador.integers(0, numero_pacientes, numero_pacientes)

        olhos = np.concatenate([indices_olhos[i] for i in sorteados])

        amostras["olho"].append(
            metricas_no_limiar(classes_olho[olhos], probs_olho[olhos], limiar_olho)
        )

        amostras["paciente"].append(
            metricas_no_limiar(
                classes_paciente[sorteados],
                probs_paciente[sorteados],
                limiar_paciente
            )
        )

    estimativas = {
        "olho": metricas_no_limiar(classes_olho, probs_olho, limiar_olho),
        "paciente": metricas_no_limiar(
            classes_paciente,
            probs_paciente,
            limiar_paciente
        )
    }

    linhas = {}

    for nivel, lista in amostras.items():
        reamostradas = pd.DataFrame(lista)

        for metrica in reamostradas.columns:
            inferior, superior = np.percentile(
                reamostradas[metrica],
                [2.5, 97.5]
            )

            linhas[(nivel, metrica)] = {
                "estimativa": estimativas[nivel][metrica],
                "ic95_inferior": inferior,
                "ic95_superior": superior
            }

    return pd.DataFrame(linhas).T
