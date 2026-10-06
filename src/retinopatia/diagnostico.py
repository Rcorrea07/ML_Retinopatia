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

O módulo também permite diagnosticar o modelo do treino v2 sem
retreinar: `preparar_dados_v2` recria, só para validação e teste, as
imagens com o pré-processamento v2 com que ele foi treinado.
"""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    roc_auc_score
)

from . import avaliacao, configuracao, pipeline_dados, pre_processamento
from .modelo import GeM


def carregar_modelo(caminho=configuracao.CAMINHO_MODELO_V2):
    """
    Carrega um modelo salvo em `.keras` (só para predição, sem
    recompilar).
    """

    if not os.path.isfile(caminho):
        raise FileNotFoundError(
            f"Modelo não encontrado em {caminho}. "
            "Suba o arquivo .keras para essa pasta do Drive."
        )

    print(f"Carregando o modelo de {caminho}...")

    return tf.keras.models.load_model(
        caminho,
        custom_objects={"GeM": GeM},
        compile=False
    )


def preparar_dados_v2(
    df_val,
    df_teste,
    pasta_imagens_v2=configuracao.PASTA_IMAGENS_V2,
    opcoes_v2=configuracao.OPCOES_PRE_PROCESSAMENTO_V2,
    pasta_diagnostico=configuracao.PASTA_DIAGNOSTICO_V2
):
    """
    Processa só as imagens de validação e teste com o pré-processamento
    v2 (Otsu, Ben sem máscara) e monta os pipelines delas.

    Precisa das fotos originais extraídas. A contagem de máscaras
    suspeitas mostra quantas imagens o Otsu da v2 recortou mal.

    Retorna:
        (val_data_v2, teste_data_v2)
    """

    pre_processamento.pre_processar_dataset(
        pd.concat([df_val, df_teste]),
        pasta_saida=pasta_imagens_v2,
        caminho_lista_suspeitas=os.path.join(
            pasta_diagnostico,
            "mascaras_suspeitas_v2_val_teste.csv"
        ),
        **opcoes_v2
    )

    val_data = pipeline_dados.criar_dataset(
        df_val,
        treino=False,
        pasta_imagens=pasta_imagens_v2
    )

    teste_data = pipeline_dados.criar_dataset(
        df_teste,
        treino=False,
        pasta_imagens=pasta_imagens_v2
    )

    return val_data, teste_data


def _metricas(classes, probabilidades, limiar):
    """
    ROC-AUC, PR-AUC, sensibilidade e especificidade no limiar.
    """

    classes = np.asarray(classes)
    preditas = np.asarray(probabilidades) >= limiar

    return {
        "roc_auc": roc_auc_score(classes, probabilidades),
        "pr_auc": average_precision_score(classes, probabilidades),
        "sensibilidade": preditas[classes == 1].mean(),
        "especificidade": 1 - preditas[classes == 0].mean()
    }


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
    grau_minimo=configuracao.GRAU_REFERENCIAVEL,
    sensibilidade_desejada=configuracao.SENSIBILIDADE_DESEJADA
):
    """
    Desempenho das mesmas probabilidades para retinopatia referenciável
    (grau >= `grau_minimo`), por olho e por paciente. Os limiares desse
    alvo são escolhidos na validação e medidos no teste.
    """

    _titulo(f"Retinopatia referenciável (grau >= {grau_minimo})")

    linhas = {}

    for nivel in ("olho", "paciente"):
        conjuntos = {}

        for nome, tabela in (("validação", tabela_val), ("teste", tabela_teste)):
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
                **_metricas(classes, probabilidades, limiar),
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

    grau_arredondado = np.clip(
        np.rint(tabela_val["grau_previsto"]),
        0,
        4
    ).astype(int)

    kappa = cohen_kappa_score(
        tabela_val["level_original"],
        grau_arredondado,
        weights="quadratic"
    )

    erro_medio = np.abs(
        tabela_val["grau_previsto"] - tabela_val["level_original"]
    ).mean()

    print(f"\nKappa quadrático do grau arredondado: {kappa:.4f}")
    print(f"Erro absoluto médio do grau: {erro_medio:.4f}")

    return linhas


def comparar_agregacao_paciente(
    tabela_val,
    agregacoes=configuracao.AGREGACOES_PACIENTE,
    sensibilidade_desejada=configuracao.SENSIBILIDADE_DESEJADA
):
    """
    Na validação: compara formas de combinar os dois olhos do paciente.
    A especificidade é a obtida no limiar que atinge a sensibilidade
    desejada (escolhido na própria validação, igual para todas).
    """

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

        linhas[funcao] = _metricas(classes, probabilidades, limiar)

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
    numero_reamostragens=configuracao.NUMERO_BOOTSTRAP,
    seed=configuracao.SEED
):
    """
    Intervalos de confiança de 95% por bootstrap, reamostrando pacientes
    (os dois olhos de um paciente entram ou saem juntos, porque não são
    independentes). Usa os limiares escolhidos na validação.
    """

    _titulo(
        f"{nome} — intervalos de confiança de 95% "
        f"({numero_reamostragens} reamostragens de pacientes)"
    )

    por_paciente = tabela.groupby("patient_id").agg(
        target=("target", "max"),
        prob_doente=("prob_doente", "max")
    )

    linhas_por_paciente = tabela.groupby("patient_id").indices

    indices_olhos = [
        linhas_por_paciente[paciente]
        for paciente in por_paciente.index
    ]

    classes_olho = tabela["target"].astype(int).values
    probs_olho = tabela["prob_doente"].values
    classes_paciente = por_paciente["target"].astype(int).values
    probs_paciente = por_paciente["prob_doente"].values

    gerador = np.random.default_rng(seed)
    numero_pacientes = len(por_paciente)

    amostras = {"olho": [], "paciente": []}

    for _ in range(numero_reamostragens):
        pacientes = gerador.integers(0, numero_pacientes, numero_pacientes)

        olhos = np.concatenate([indices_olhos[i] for i in pacientes])

        amostras["olho"].append(
            _metricas(classes_olho[olhos], probs_olho[olhos], limiar_olho)
        )

        amostras["paciente"].append(
            _metricas(
                classes_paciente[pacientes],
                probs_paciente[pacientes],
                limiar_paciente
            )
        )

    estimativas = {
        "olho": _metricas(classes_olho, probs_olho, limiar_olho),
        "paciente": _metricas(
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

    resultado = pd.DataFrame(linhas).T

    print(resultado.round(4).to_string())

    return resultado


def listar_falsos_negativos(
    tabela,
    limiar,
    grau_minimo=configuracao.GRAU_REFERENCIAVEL,
    pasta_imagens=configuracao.PASTA_IMAGENS_OTIMIZADAS,
    numero_imagens=configuracao.NUMERO_IMAGENS_ERRO,
    nome="Teste"
):
    """
    Lista os olhos de grau >= `grau_minimo` que o modelo deixou passar
    no limiar por olho e mostra as imagens processadas (o que o modelo
    viu), das mais confiantes no erro para as menos.
    """

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
    resultado_teste,
    pasta_imagens=configuracao.PASTA_IMAGENS_OTIMIZADAS
):
    """
    Executa todo o diagnóstico a partir dos resultados de
    `avaliacao.avaliar_modelo` (validação) e `avaliacao.avaliar_no_teste`
    (teste), usando os limiares escolhidos na validação.

    Retorna um dicionário com as tabelas de cada parte.
    """

    tabela_val = resultado_validacao["tabela"]
    tabela_teste = resultado_teste["tabela"]

    limiar_olho = resultado_validacao["limiar_olho"]
    limiar_paciente = resultado_validacao["limiar_paciente"]

    return {
        "por_grau": relatorio_por_grau(tabela_teste, limiar_olho),
        "referenciavel": avaliar_referenciavel(tabela_val, tabela_teste),
        "cabeca_grau": comparar_cabeca_grau(tabela_val),
        "agregacao": comparar_agregacao_paciente(tabela_val),
        "bootstrap": intervalo_bootstrap(
            tabela_teste,
            limiar_olho,
            limiar_paciente
        ),
        "falsos_negativos": listar_falsos_negativos(
            tabela_teste,
            limiar_olho,
            pasta_imagens=pasta_imagens
        )
    }
