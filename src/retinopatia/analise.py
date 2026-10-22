"""
Análises dos treinos já feitos, a partir das predições salvas em CSV.

Roda em CPU e não importa TensorFlow: lê os arquivos de
`resultados/<experimento>/`, sem treinar nem prever de novo. Serve para:

- comparar todas as execuções na validação, com e sem TTA
  (`tabela_comparativa`);
- medir o que se ganharia combinando os dois olhos com um modelo
  aprendido em vez do máximo (`combinador_paciente`), combinando as
  cabeças `doente` e `grau` (`misturar_cabecas`) ou fazendo ensemble de
  execuções (`ensemble`);
- calcular o kappa quadrático da cabeça de grau (`kappa_grau`), a métrica
  das competições do Kaggle.

Regras iguais às da avaliação (`avaliacao`): o paciente é doente se algum
olho for, a probabilidade dele é a maior dos dois olhos, e o limiar é o
de menor taxa de falsos positivos com sensibilidade de pelo menos 0,90.

Tudo aqui é medido na validação. Quando algo é ajustado (pesos de uma
regressão, mistura das cabeças), o ajuste é feito por validação cruzada
entre pacientes da própria validação, para não medir no mesmo dado em
que se ajustou. O ensemble de várias redes serve para medir o teto e
escolher professor: o modelo do celular é uma rede só.
"""

import os
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import StratifiedKFold

from . import configuracao

# Modos de TTA que podem aparecer nas predições salvas
MODOS = ["sem_tta", "tta2", "tta4"]

NUMERO_DOBRAS = 5


# ============================================================
# LEITURA DAS PREDIÇÕES
# ============================================================


def _tabela_modo(tabela, prob, grau):
    """
    Tabela padronizada de um modo: image, patient_id, level_original,
    target, prob, grau.
    """

    return pd.DataFrame({
        "image": tabela["image"].values,
        "patient_id": tabela["patient_id"].values,
        "level_original": tabela["level_original"].astype(int).values,
        "target": tabela["target"].astype(int).values,
        "prob": tabela[prob].astype(float).values,
        "grau": tabela[grau].astype(float).values
    })


def carregar_predicoes(pasta):
    """
    Predições de validação da Fase 2 de uma execução, por modo de TTA.

    Entende os três formatos que existem no Drive:
    - atual: `predicoes_validacao_fase_2.csv` com as colunas
      `prob_doente_<modo>` e `grau_previsto_<modo>`;
    - campanha de outubro: `predicoes_validacao_fase_2_sem_tta.csv` e
      `predicoes_validacao_fase_2_tta4.csv`;
    - antigo: só `predicoes_validacao_fase_2.csv` com `prob_doente`
      (avaliado com TTA de 4 variantes).

    Retorna:
        {modo: tabela}; vazio se a execução não tem predições da Fase 2
    """

    resultado = {}

    caminho = os.path.join(pasta, "predicoes_validacao_fase_2.csv")

    if os.path.isfile(caminho):
        tabela = pd.read_csv(caminho)

        modos_salvos = [m for m in MODOS if f"prob_doente_{m}" in tabela]

        if modos_salvos:
            for modo in modos_salvos:
                resultado[modo] = _tabela_modo(
                    tabela,
                    f"prob_doente_{modo}",
                    f"grau_previsto_{modo}"
                )
        else:
            resultado["tta4"] = _tabela_modo(
                tabela,
                "prob_doente",
                "grau_previsto"
            )

    for modo in ("sem_tta", "tta4"):
        caminho = os.path.join(pasta, f"predicoes_validacao_fase_2_{modo}.csv")

        if modo not in resultado and os.path.isfile(caminho):
            resultado[modo] = _tabela_modo(
                pd.read_csv(caminho),
                "prob_doente",
                "grau_previsto"
            )

    return resultado


def listar_execucoes(pasta_resultados=configuracao.PASTA_RESULTADOS_GERAL):
    """
    Pastas de resultados com predições de validação da Fase 2, sem as
    execuções de fumaça.

    Retorna:
        {nome da pasta: {modo: tabela}}
    """

    execucoes = {}

    for nome in sorted(os.listdir(pasta_resultados)):
        pasta = os.path.join(pasta_resultados, nome)

        if not os.path.isdir(pasta) or nome.endswith("_fumaca"):
            continue

        predicoes = carregar_predicoes(pasta)

        if predicoes:
            execucoes[nome] = predicoes

    return execucoes


# ============================================================
# MÉTRICAS
# ============================================================


def _especificidade_na_sensibilidade(
    classes,
    probabilidades,
    sensibilidade_desejada=configuracao.SENSIBILIDADE_DESEJADA
):
    """
    Especificidade e limiar no ponto da curva ROC com sensibilidade de
    pelo menos `sensibilidade_desejada` e menor taxa de falsos positivos
    (a mesma regra de `avaliacao.escolher_limiar`).
    """

    fpr, tpr, limiares = roc_curve(classes, probabilidades)

    validos = np.where(tpr >= sensibilidade_desejada)[0]

    if len(validos) == 0:
        return np.nan, np.nan

    melhor = validos[np.argmin(fpr[validos])]

    return 1 - fpr[melhor], limiares[melhor]


def por_paciente(tabela, coluna="prob"):
    """
    Uma linha por paciente: classe (doente se algum olho for) e a maior
    probabilidade entre os olhos.
    """

    return tabela.groupby("patient_id").agg(
        target=("target", "max"),
        prob=(coluna, "max")
    )


def metricas(tabela, coluna="prob"):
    """
    ROC-AUC, AP e especificidade (sensibilidade ≥ 0,90) por olho e por
    paciente.
    """

    resultado = {}

    pacientes = por_paciente(tabela, coluna)

    for nivel, classes, probabilidades in (
        ("olho", tabela["target"], tabela[coluna]),
        ("paciente", pacientes["target"], pacientes["prob"])
    ):
        especificidade, _ = _especificidade_na_sensibilidade(
            classes,
            probabilidades
        )

        resultado[f"{nivel}_roc_auc"] = roc_auc_score(classes, probabilidades)
        resultado[f"{nivel}_ap"] = average_precision_score(
            classes,
            probabilidades
        )
        resultado[f"{nivel}_especificidade"] = especificidade

    return resultado


def kappa_grau(tabela):
    """
    Kappa quadrático da cabeça de grau (arredondada e limitada a 0–4),
    por olho. Referência: 0,85 do 1º lugar do Kaggle 2015, medido no
    teste da competição (outra amostra; serve de ordem de grandeza).
    """

    previstos = np.clip(np.rint(tabela["grau"]), 0, 4).astype(int)

    return cohen_kappa_score(
        tabela["level_original"],
        previstos,
        weights="quadratic"
    )


def tabela_comparativa(
    execucoes=None,
    referencia=r"^E01_"
):
    """
    Uma linha por execução e modo de TTA, com as métricas de validação e
    o kappa da cabeça de grau, ordenada pela especificidade por paciente.
    As execuções cujo nome casa com `referencia` são marcadas.

    `execucoes` é o resultado de `listar_execucoes` (lido do Drive se
    for None).
    """

    if execucoes is None:
        execucoes = listar_execucoes()

    linhas = []

    for nome, modos in execucoes.items():
        for modo, tabela in modos.items():
            linhas.append({
                "execucao": nome,
                "modo": modo,
                "referencia": bool(re.search(referencia, nome)),
                **metricas(tabela),
                "kappa_grau": kappa_grau(tabela)
            })

    if not linhas:
        return pd.DataFrame()

    return pd.DataFrame(linhas).sort_values(
        "paciente_especificidade",
        ascending=False
    ).reset_index(drop=True)


# ============================================================
# COMBINAÇÕES (ajustadas por validação cruzada entre pacientes)
# ============================================================


def _caracteristicas_paciente(tabela):
    """
    Características do par de olhos de cada paciente, sem depender de
    qual olho é o esquerdo: maior e menor probabilidade, maior e menor
    grau previsto e a diferença entre os olhos.
    """

    agrupado = tabela.groupby("patient_id")

    caracteristicas = pd.DataFrame({
        "prob_max": agrupado["prob"].max(),
        "prob_min": agrupado["prob"].min(),
        "grau_max": agrupado["grau"].max(),
        "grau_min": agrupado["grau"].min()
    })

    caracteristicas["prob_diferenca"] = (
        caracteristicas["prob_max"] - caracteristicas["prob_min"]
    )

    return caracteristicas, agrupado["target"].max()


def _fora_da_dobra(caracteristicas, classes, ajustar, seed=configuracao.SEED_SPLIT):
    """
    Predições de cada paciente feitas por um ajuste que não o viu:
    validação cruzada estratificada em NUMERO_DOBRAS dobras de pacientes.
    `ajustar(x_treino, y_treino, x_teste)` devolve as predições do teste.
    """

    predicoes = np.zeros(len(classes))

    dobras = StratifiedKFold(
        n_splits=NUMERO_DOBRAS,
        shuffle=True,
        random_state=seed
    )

    for treino, teste in dobras.split(caracteristicas, classes):
        predicoes[teste] = ajustar(
            caracteristicas[treino],
            classes[treino],
            caracteristicas[teste]
        )

    return predicoes


def _comparar_paciente(classes, alternativas):
    """
    Tabela com ROC-AUC, AP e especificidade por paciente de cada
    alternativa ({nome: probabilidades por paciente}).
    """

    linhas = []

    for nome, probabilidades in alternativas.items():
        especificidade, _ = _especificidade_na_sensibilidade(
            classes,
            probabilidades
        )

        linhas.append({
            "regra": nome,
            "roc_auc": roc_auc_score(classes, probabilidades),
            "ap": average_precision_score(classes, probabilidades),
            "especificidade": especificidade
        })

    return pd.DataFrame(linhas).set_index("regra")


def combinador_paciente(tabela):
    """
    Compara a regra do máximo com uma regressão logística sobre as
    características dos dois olhos (`_caracteristicas_paciente`),
    avaliada fora da dobra. Mostra quanto um combinador aprendido, como
    o blend por paciente do time o_O, acrescenta ao máximo.

    Retorna a tabela de comparação por paciente.
    """

    caracteristicas, classes = _caracteristicas_paciente(tabela)

    x = caracteristicas.values
    y = classes.values

    def ajustar(x_treino, y_treino, x_teste):
        modelo = LogisticRegression(max_iter=1000)
        modelo.fit(x_treino, y_treino)
        return modelo.predict_proba(x_teste)[:, 1]

    return _comparar_paciente(y, {
        "maximo": caracteristicas["prob_max"].values,
        "logistica_dois_olhos": _fora_da_dobra(x, y, ajustar)
    })


def misturar_cabecas(tabela, pesos=np.linspace(0, 1, 11)):
    """
    Compara a probabilidade da cabeça `doente` com a mistura
    (1 - w) * prob + w * grau/4, por paciente (máximo dos olhos). O peso
    w é escolhido nas dobras de ajuste e medido na dobra que ficou de
    fora.

    Retorna a tabela de comparação por paciente.
    """

    caracteristicas, classes = _caracteristicas_paciente(tabela)

    x = caracteristicas[["prob_max", "grau_max"]].values
    y = classes.values

    def misturar(x_parte, peso):
        return (1 - peso) * x_parte[:, 0] + peso * np.clip(x_parte[:, 1], 0, 4) / 4

    def ajustar(x_treino, y_treino, x_teste):
        melhor = max(
            pesos,
            key=lambda peso: np.nan_to_num(
                _especificidade_na_sensibilidade(
                    y_treino,
                    misturar(x_treino, peso)
                )[0],
                nan=-1
            )
        )
        return misturar(x_teste, melhor)

    return _comparar_paciente(y, {
        "doente": x[:, 0],
        "mistura_doente_grau": _fora_da_dobra(x, y, ajustar)
    })


def ensemble(tabelas):
    """
    Média das probabilidades e dos graus de várias execuções, alinhadas
    pela imagem. Mede o teto de um ensemble (não é o modelo do celular).

    Retorna uma tabela no mesmo formato das de `carregar_predicoes`.
    """

    tabelas = list(tabelas)

    base = tabelas[0][["image", "patient_id", "level_original", "target"]]

    probabilidades = []
    graus = []

    for tabela in tabelas:
        alinhada = base[["image"]].merge(
            tabela[["image", "prob", "grau"]],
            on="image",
            how="left",
            validate="one_to_one"
        )

        if alinhada["prob"].isna().any():
            raise ValueError("As execuções não têm as mesmas imagens de validação.")

        probabilidades.append(alinhada["prob"].values)
        graus.append(alinhada["grau"].values)

    resultado = base.copy()
    resultado["prob"] = np.mean(probabilidades, axis=0)
    resultado["grau"] = np.mean(graus, axis=0)

    return resultado
