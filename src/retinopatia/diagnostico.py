"""
Diagnóstico do desempenho de um modelo já treinado.

A avaliação (`avaliacao`) responde "quanto o modelo acerta". Este módulo
responde "onde ele erra e quanto dá para confiar nos números":

- sensibilidade por grau original (o alvo junta o grau 1, o mais difícil
  e mais ruidoso, com os graus graves);
- desempenho para retinopatia referenciável (grau 2 ou mais), usando as
  mesmas probabilidades do modelo;
- se a cabeça auxiliar de grau traz informação além da cabeça binária;
- outras formas de combinar os dois olhos de um paciente;
- intervalos de confiança por bootstrap, reamostrando pacientes;
- a lista (e as imagens) dos falsos negativos de grau 2 ou mais.

Tudo parte das tabelas de predições por imagem devolvidas por
`avaliacao.avaliar_modelo` e `avaliacao.avaliar_no_teste` (chave
"tabela"). As comparações de alternativas (cabeça de grau, agregação)
usam só a validação: o teste continua sendo só medido.

Não usa TensorFlow: trabalha só com as tabelas de predições.
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import avaliacao, configuracao, metricas


def _titulo(texto):
    print(f"\n{'=' * 60}\n{texto}\n{'=' * 60}")


def relatorio_por_grau(tabela, limiar, nome="Teste"):
    """
    Fração de olhos marcados como doentes, por grau original, no limiar
    por olho. No grau 0 é a taxa de falsos positivos; nos graus 1 a 4,
    a sensibilidade daquele grau.
    """

    _titulo(f"{nome} — olhos marcados como doentes por grau (limiar {limiar:.4f})")

    marcados = tabela["prob_doente"] >= limiar

    por_grau = (
        tabela
        .assign(marcado=marcados)
        .groupby("level_original")
        .agg(
            olhos=("marcado", "size"),
            marcados=("marcado", "sum"),
            fracao_marcados=("marcado", "mean"),
            prob_mediana=("prob_doente", "median")
        )
    )

    print(por_grau.round(3).to_string())

    print(
        "\nGrau 0: fração = falsos positivos. "
        "Graus 1 a 4: fração = sensibilidade do grau."
    )

    return por_grau


def avaliar_referenciavel(
    tabela_val,
    tabela_teste,
    grau_minimo=None,
    sensibilidade_desejada=None,
    nome_medido="teste"
):
    """
    Desempenho das mesmas probabilidades para retinopatia referenciável
    (grau >= `grau_minimo`), por olho e por paciente. Os limiares desse
    alvo são escolhidos na validação e medidos em `tabela_teste` (o
    teste, ou uma base de validação externa: `nome_medido`).
    """

    if grau_minimo is None:
        grau_minimo = configuracao.GRAU_REFERENCIAVEL

    if sensibilidade_desejada is None:
        sensibilidade_desejada = configuracao.SENSIBILIDADE_DESEJADA

    _titulo(f"Retinopatia referenciável (grau >= {grau_minimo})")

    linhas = {}

    for nivel in ("olho", "paciente"):
        conjuntos = {}

        for nome, tabela in (("validação", tabela_val), (nome_medido, tabela_teste)):
            tabela = tabela.assign(
                target=(tabela["level_original"] >= grau_minimo).astype(int)
            )

            if nivel == "olho":
                conjuntos[nome] = (
                    tabela["target"].values,
                    tabela["prob_doente"].values
                )

            else:
                conjuntos[nome] = avaliacao.agregar_por_paciente(
                    tabela,
                    tabela["prob_doente"].values
                )

        limiar, _, _ = avaliacao.escolher_limiar(
            *conjuntos["validação"],
            sensibilidade_desejada
        )

        for nome, (classes, probabilidades) in conjuntos.items():
            linhas[(nivel, nome)] = {
                **metricas.metricas_no_limiar(classes, probabilidades, limiar),
                "limiar": limiar
            }

    resultado = pd.DataFrame(linhas).T

    print(resultado.round(4).to_string())

    print(
        "\nOs limiares deste alvo foram escolhidos na validação; "
        "a linha do teste é a estimativa honesta."
    )

    return resultado


def comparar_cabeca_grau(tabela_val):
    """
    Na validação: a cabeça de grau separa doentes de saudáveis tão bem
    quanto a cabeça binária? A média dos postos (ranks) das duas ajuda?
    Mostra também o kappa quadrático do grau arredondado (a métrica das
    competições do Kaggle) e o erro absoluto médio.
    """

    _titulo("Cabeça de grau x cabeça binária (validação)")

    escores = {
        "doente (binária)": tabela_val["prob_doente"],
        "grau (regressão)": tabela_val["grau_previsto"],
        "média dos postos": (
            tabela_val["prob_doente"].rank(pct=True)
            + tabela_val["grau_previsto"].rank(pct=True)
        ) / 2
    }

    alvos = {
        "grau >= 1": tabela_val["target"],
        f"grau >= {configuracao.GRAU_REFERENCIAVEL}": (
            tabela_val["level_original"] >= configuracao.GRAU_REFERENCIAVEL
        ).astype(int)
    }

    linhas = {
        nome_escore: {
            f"ROC-AUC {nome_alvo}": roc_auc_score(alvo, escore)
            for nome_alvo, alvo in alvos.items()
        }
        for nome_escore, escore in escores.items()
    }

    print(pd.DataFrame(linhas).T.round(4).to_string())

    kappa = metricas.kappa_grau(
        tabela_val["level_original"],
        tabela_val["grau_previsto"]
    )

    erro_medio = np.abs(
        tabela_val["grau_previsto"] - tabela_val["level_original"]
    ).mean()

    print(f"\nKappa quadrático do grau arredondado: {kappa:.4f}")
    print(f"Erro absoluto médio do grau: {erro_medio:.4f}")

    return linhas


def comparar_agregacao_paciente(
    tabela_val,
    agregacoes=None,
    sensibilidade_desejada=None
):
    """
    Na validação: compara formas de combinar os dois olhos do paciente.
    A especificidade é a obtida no limiar que atinge a sensibilidade
    desejada (escolhido na própria validação, igual para todas).
    """

    if agregacoes is None:
        agregacoes = configuracao.AGREGACOES_PACIENTE

    if sensibilidade_desejada is None:
        sensibilidade_desejada = configuracao.SENSIBILIDADE_DESEJADA

    _titulo("Agregação dos dois olhos por paciente (validação)")

    linhas = {}

    for funcao in agregacoes:
        classes, probabilidades = avaliacao.agregar_por_paciente(
            tabela_val,
            tabela_val["prob_doente"].values,
            funcao
        )

        limiar, _, _ = avaliacao.escolher_limiar(
            classes,
            probabilidades,
            sensibilidade_desejada
        )

        linhas[funcao] = metricas.metricas_no_limiar(classes, probabilidades, limiar)

    resultado = pd.DataFrame(linhas).T

    print(resultado.round(4).to_string())

    print(
        "\nUma diferença só vale se for maior que o ruído "
        "(veja os intervalos do bootstrap)."
    )

    return resultado


def intervalo_bootstrap(
    tabela,
    limiar_olho,
    limiar_paciente,
    nome="Teste",
    numero_reamostragens=None,
    seed=None
):
    """
    Intervalos de confiança de 95% por bootstrap, reamostrando pacientes
    (os dois olhos de um paciente entram ou saem juntos, porque não são
    independentes). Usa os limiares escolhidos na validação.
    """

    if numero_reamostragens is None:
        numero_reamostragens = configuracao.NUMERO_BOOTSTRAP

    if seed is None:
        seed = configuracao.SEED_TREINO

    _titulo(
        f"{nome} — intervalos de confiança de 95% "
        f"({numero_reamostragens} reamostragens de pacientes)"
    )

    resultado = metricas.intervalos_bootstrap(
        tabela,
        limiar_olho,
        limiar_paciente,
        numero_reamostragens,
        seed
    )

    print(resultado.round(4).to_string())

    return resultado


def listar_falsos_negativos(
    tabela,
    limiar,
    grau_minimo=None,
    pasta_imagens=None,
    numero_imagens=None,
    nome="Teste"
):
    """
    Lista os olhos de grau >= `grau_minimo` que o modelo deixou passar
    no limiar por olho e mostra as imagens processadas (o que o modelo
    viu), das mais confiantes no erro para as menos.
    """

    if grau_minimo is None:
        grau_minimo = configuracao.GRAU_REFERENCIAVEL

    if pasta_imagens is None:
        pasta_imagens = configuracao.PASTA_IMAGENS_OTIMIZADAS

    if numero_imagens is None:
        numero_imagens = configuracao.NUMERO_IMAGENS_ERRO

    _titulo(
        f"{nome} — falsos negativos de grau >= {grau_minimo} "
        f"(limiar {limiar:.4f})"
    )

    falsos_negativos = tabela[
        (tabela["level_original"] >= grau_minimo)
        & (tabela["prob_doente"] < limiar)
    ].sort_values("prob_doente")

    total = (tabela["level_original"] >= grau_minimo).sum()

    print(
        f"{len(falsos_negativos)} de {total} olhos de grau >= "
        f"{grau_minimo} não foram detectados."
    )

    print(
        falsos_negativos[
            ["image", "level_original", "prob_doente", "grau_previsto"]
        ]
        .round(4)
        .to_string(index=False)
    )

    exemplos = falsos_negativos.head(numero_imagens)

    if exemplos.empty:
        return falsos_negativos

    colunas = 4
    linhas = int(np.ceil(len(exemplos) / colunas))

    plt.figure(figsize=(4 * colunas, 4 * linhas))

    for posicao, (_, linha) in enumerate(exemplos.iterrows(), start=1):
        img = cv2.imread(os.path.join(pasta_imagens, linha["image"]))

        plt.subplot(linhas, colunas, posicao)

        if img is not None:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        plt.title(
            f"{linha['image']}\n"
            f"grau {linha['level_original']} | "
            f"p = {linha['prob_doente']:.3f}"
        )
        plt.axis("off")

    plt.tight_layout()
    plt.show()

    return falsos_negativos


def relatorio_completo(
    resultado_validacao,
    resultado_teste=None,
    pasta_imagens=None,
    pasta_resultados=None
):
    """
    Executa todo o diagnóstico a partir dos resultados de
    `avaliacao.avaliar_modelo` (validação) e, para o modelo final,
    `avaliacao.avaliar_no_teste` (teste), usando os limiares escolhidos
    na validação.

    Sem `resultado_teste` (experimentos), as partes medidas no teste
    passam a ser medidas na própria validação: servem para comparar
    experimentos, mas são otimistas.

    As tabelas também são gravadas em CSV na pasta do experimento.
    Retorna um dicionário com as tabelas de cada parte.
    """

    if pasta_imagens is None:
        pasta_imagens = configuracao.PASTA_IMAGENS_OTIMIZADAS

    if pasta_resultados is None:
        pasta_resultados = configuracao.PASTA_RESULTADOS

    tabela_val = resultado_validacao["tabela"]

    if resultado_teste is None:
        tabela_medida, nome = tabela_val, "Validação"
    else:
        tabela_medida, nome = resultado_teste["tabela"], "Teste"

    limiar_olho = resultado_validacao["limiar_olho"]
    limiar_paciente = resultado_validacao["limiar_paciente"]

    partes = {
        "por_grau": relatorio_por_grau(tabela_medida, limiar_olho, nome),
        "referenciavel": avaliar_referenciavel(tabela_val, tabela_medida),
        "cabeca_grau": comparar_cabeca_grau(tabela_val),
        "agregacao": comparar_agregacao_paciente(tabela_val),
        "bootstrap": intervalo_bootstrap(
            tabela_medida,
            limiar_olho,
            limiar_paciente,
            nome
        ),
        "falsos_negativos": listar_falsos_negativos(
            tabela_medida,
            limiar_olho,
            pasta_imagens=pasta_imagens,
            nome=nome
        )
    }

    os.makedirs(pasta_resultados, exist_ok=True)

    for parte, tabela in partes.items():
        if isinstance(tabela, pd.DataFrame):
            tabela.to_csv(os.path.join(
                pasta_resultados,
                f"diagnostico_{parte}_{'validacao' if resultado_teste is None else 'teste'}.csv"
            ))

    return partes
