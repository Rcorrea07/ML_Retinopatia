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

O módulo tem três camadas:

- predição (`prever_variantes`): a única que usa TensorFlow, importado
  só dentro das funções;
- medida (`tabela_predicoes`, `comparar_modos`, `medir_predicoes`):
  números a partir das predições, sem `print` nem gráficos. Serve a
  qualquer conjunto (validação, teste, validação externa) e a modelos
  de fora do Keras, desde que entreguem as predições no mesmo formato;
- exibição (`mostrar_avaliacao`, `plotar_*`).

`avaliar_predicoes` junta as três depois da predição; `avaliar_modelo`
(validação, nas duas fases de treino) e `avaliar_no_teste` (modelo
final) são ela mais a predição. As predições de cada imagem são salvas
em CSV na pasta de resultados, para que o diagnóstico possa ser refeito
sem rodar o modelo de novo.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from . import configuracao, metricas
from .metricas import kappa_grau  # noqa: F401 (parte da interface da avaliação)

# Modos da test-time augmentation e quantas variantes cada um usa: "tta2"
# é a média das duas primeiras (original e espelho horizontal) e "tta4" a
# média das quatro. No celular, cada variante é uma passagem a mais pela
# rede.
MODOS_TTA = {
    "sem_tta": 1,
    "tta2": 2,
    "tta4": 4
}


# ============================================================
# HISTÓRICO DO TREINO
# ============================================================


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


# ============================================================
# PREDIÇÃO (TensorFlow)
# ============================================================


def _variantes_tta():
    """
    Transformações da test-time augmentation, na ordem em que são
    previstas: original, espelho horizontal, espelho vertical e os dois.
    """

    import tensorflow as tf

    return [
        lambda x: x,
        tf.image.flip_left_right,
        tf.image.flip_up_down,
        lambda x: tf.image.flip_up_down(tf.image.flip_left_right(x))
    ]


def _prever_variante(modelo, dados, transformar):
    """
    Probabilidades e graus de uma variante, na ordem do pipeline (que
    não embaralha). No modelo de pares a entrada é um dicionário com os
    dois olhos, e a transformação vale para os dois.
    """

    import tensorflow as tf

    # O pipeline devolve (imagens, rótulos); só as imagens interessam
    imagens = dados.map(
        lambda x, *_, f=transformar: tf.nest.map_structure(f, x)
    )

    saida = modelo.predict(imagens, verbose=0)

    # Um modelo carregado do disco pode devolver lista em vez de
    # dicionário; os nomes das saídas dizem a ordem
    if not isinstance(saida, dict):
        saida = dict(zip(modelo.output_names, saida))

    return (
        np.asarray(saida[configuracao.SAIDA_DOENTE]).ravel(),
        np.asarray(saida[configuracao.SAIDA_GRAU]).ravel()
    )


def prever_variantes(modelo, dados):
    """
    Prevê as quatro variantes da TTA uma vez e monta, a partir delas, as
    predições sem TTA, com TTA de 2 e com TTA de 4 variantes.

    Retorna:
        {modo: (probabilidades, graus_previstos)}, com os modos de
        MODOS_TTA
    """

    variantes = [
        _prever_variante(modelo, dados, transformar)
        for transformar in _variantes_tta()
    ]

    return {
        modo: (
            np.mean([v[0] for v in variantes[:quantidade]], axis=0),
            np.mean([v[1] for v in variantes[:quantidade]], axis=0)
        )
        for modo, quantidade in MODOS_TTA.items()
    }


# ============================================================
# MEDIDA (sem TensorFlow, sem print)
# ============================================================


def agregar_por_paciente(df, probabilidades, funcao="max"):
    """
    Combina os dois olhos de cada paciente: a classe do paciente é 1 se
    qualquer olho estiver doente, e a probabilidade é, por padrão, a
    maior entre as dos dois olhos (`metricas.agregar_olhos` lista as
    alternativas).

    Retorna:
        (classes_paciente, probabilidades_paciente)
    """

    pacientes = metricas.por_paciente(
        df["patient_id"].values,
        df["target"].values,
        probabilidades,
        funcao
    )

    return pacientes["target"].values, pacientes["valor"].values


def escolher_limiar(
    classes_verdadeiras,
    predicoes,
    sensibilidade_desejada=None
):
    """
    Escolhe, na curva ROC, o limiar que atinge pelo menos a
    sensibilidade desejada (padrão: `SENSIBILIDADE_DESEJADA`) com a
    menor taxa de falsos positivos.

    Retorna:
        (limiar, sensibilidade_obtida, especificidade_obtida)

    Se nenhum limiar atingir a sensibilidade desejada, devolve o limiar
    padrão de 0.5 e None nas outras duas posições.
    """

    if sensibilidade_desejada is None:
        sensibilidade_desejada = configuracao.SENSIBILIDADE_DESEJADA

    limiar, sensibilidade, especificidade = metricas.ponto_na_sensibilidade(
        classes_verdadeiras,
        predicoes,
        sensibilidade_desejada
    )

    if limiar is None:
        return 0.5, None, None

    return limiar, sensibilidade, especificidade


def tabela_predicoes(df, variantes, modo_tta):
    """
    Tabela de predições por imagem: image, patient_id, level_original,
    target, `prob_doente` e `grau_previsto` do modo principal e as
    colunas `prob_doente_<modo>` e `grau_previsto_<modo>` de cada modo de
    `variantes`.
    """

    tabela = df[
        ["image", "patient_id", "level_original", "target"]
    ].copy()

    tabela["prob_doente"], tabela["grau_previsto"] = variantes[modo_tta]

    for modo, (probabilidades, graus) in variantes.items():
        tabela[f"prob_doente_{modo}"] = probabilidades
        tabela[f"grau_previsto_{modo}"] = graus

    return tabela


def variantes_da_tabela(tabela):
    """
    O inverso de `tabela_predicoes`: as predições por modo de TTA de uma
    tabela salva (colunas `prob_doente_<modo>` e `grau_previsto_<modo>`),
    no formato de `prever_variantes`.
    """

    variantes = {
        modo: (
            tabela[f"prob_doente_{modo}"].values,
            tabela[f"grau_previsto_{modo}"].values
        )
        for modo in MODOS_TTA
        if f"prob_doente_{modo}" in tabela
    }

    if not variantes:
        raise ValueError("A tabela não tem predições por modo de TTA (prob_doente_<modo>).")

    return variantes


def _medir_nivel(classes, probabilidades, limiar, titulo):
    """
    ROC-AUC, PR-AUC, sensibilidade e especificidade no limiar e a matriz
    de confusão de um nível (olho ou paciente).
    """

    if len(np.unique(classes)) != 2:
        raise ValueError(
            f"{titulo}: o conjunto precisa possuir as classes 0 e 1."
        )

    matriz = confusion_matrix(
        classes,
        (probabilidades >= limiar).astype(int),
        labels=[0, 1]
    )

    vn, fp, fn, vp = matriz.ravel()

    return {
        "roc_auc": roc_auc_score(classes, probabilidades),
        "pr_auc": average_precision_score(classes, probabilidades),
        "sensibilidade": vp / (vp + fn),
        "especificidade": vn / (vn + fp),
        "limiar": limiar,
        "matriz_confusao": matriz
    }


def comparar_modos(df, variantes, limiares=None, sensibilidade_desejada=None):
    """
    Compara, por paciente, as predições sem TTA, com TTA de 2 e de 4
    variantes: ROC-AUC, AP, sensibilidade e especificidade.

    Sem `limiares`, cada modo tem o seu limiar escolhido na própria
    tabela (validação; coluna `limiar`): um limiar escolhido com TTA não
    vale para a inferência sem TTA, e vice-versa. Com `limiares`
    ({modo: limiar}, os que a validação escolheu), eles são só aplicados
    (teste ou validação externa; coluna `limiar_validacao`).

    Retorna um DataFrame com uma linha por modo.
    """

    if sensibilidade_desejada is None:
        sensibilidade_desejada = configuracao.SENSIBILIDADE_DESEJADA

    linhas = []

    for modo, (probabilidades, _) in variantes.items():
        classes, probabilidades_paciente = agregar_por_paciente(
            df,
            probabilidades
        )

        linha = {
            "modo": modo,
            "passagens": MODOS_TTA[modo],
            "roc_auc": roc_auc_score(classes, probabilidades_paciente),
            "pr_auc": average_precision_score(
                classes,
                probabilidades_paciente
            )
        }

        if limiares is None:
            limiar, sensibilidade, especificidade = escolher_limiar(
                classes,
                probabilidades_paciente,
                sensibilidade_desejada
            )

            linha.update(
                sensibilidade=sensibilidade,
                especificidade=especificidade,
                limiar=limiar
            )

        else:
            limiar = limiares[modo]
            preditas = probabilidades_paciente >= limiar

            linha.update(
                sensibilidade=preditas[classes == 1].mean(),
                especificidade=1 - preditas[classes == 0].mean(),
                limiar_validacao=limiar
            )

        linhas.append(linha)

    return pd.DataFrame(linhas).set_index("modo")


def limiares_da_validacao(resultado_validacao):
    """
    Os limiares que a validação escolheu, no formato de
    `medir_predicoes`: por olho e por paciente no modo principal e, por
    paciente, de cada modo de TTA.
    """

    return {
        "olho": resultado_validacao["limiar_olho"],
        "paciente": resultado_validacao["limiar_paciente"],
        "modos": resultado_validacao["comparacao_tta"]["limiar"]
    }


def medir_predicoes(
    df,
    variantes,
    modo_tta,
    limiares=None,
    sensibilidade_desejada=None
):
    """
    Tudo o que a avaliação mede num conjunto, a partir das predições
    (`variantes`, de `prever_variantes`, alinhadas com `df`).

    Sem `limiares`, os limiares por olho e por paciente são escolhidos
    aqui (validação). Com `limiares` (`limiares_da_validacao`), eles só
    são aplicados: é a medida honesta do teste ou de uma validação
    externa.

    Retorna um dicionário com o modo de TTA, os limiares, as métricas
    por olho e por paciente, o kappa da cabeça de grau, a comparação dos
    modos de TTA e as predições do modo principal.
    """

    if sensibilidade_desejada is None:
        sensibilidade_desejada = configuracao.SENSIBILIDADE_DESEJADA

    predicoes, graus_previstos = variantes[modo_tta]

    classes_olho = df["target"].astype(int).values

    classes_paciente, predicoes_paciente = agregar_por_paciente(
        df,
        predicoes
    )

    sem_ponto = []

    if limiares is None:
        limiar_olho, sensibilidade_olho, _ = escolher_limiar(
            classes_olho,
            predicoes,
            sensibilidade_desejada
        )

        limiar_paciente, sensibilidade_paciente, _ = escolher_limiar(
            classes_paciente,
            predicoes_paciente,
            sensibilidade_desejada
        )

        sem_ponto = [
            nivel
            for nivel, sensibilidade in (
                ("olho", sensibilidade_olho),
                ("paciente", sensibilidade_paciente)
            )
            if sensibilidade is None
        ]

        comparacao_tta = comparar_modos(
            df,
            variantes,
            sensibilidade_desejada=sensibilidade_desejada
        )

    else:
        limiar_olho = limiares["olho"]
        limiar_paciente = limiares["paciente"]

        comparacao_tta = comparar_modos(
            df,
            variantes,
            limiares=limiares["modos"]
        )

    return {
        "modo_tta": modo_tta,
        "limiares_escolhidos": limiares is None,
        "sem_ponto": sem_ponto,
        "limiar_olho": limiar_olho,
        "limiar_paciente": limiar_paciente,
        "metricas_olho": _medir_nivel(
            classes_olho,
            predicoes,
            limiar_olho,
            "por olho"
        ),
        "metricas_paciente": _medir_nivel(
            classes_paciente,
            predicoes_paciente,
            limiar_paciente,
            "por paciente"
        ),
        "kappa_grau": kappa_grau(df["level_original"], graus_previstos),
        "comparacao_tta": comparacao_tta,
        "predicoes": predicoes,
        "graus_previstos": graus_previstos
    }


# ============================================================
# EXIBIÇÃO
# ============================================================


def plotar_matriz_confusao(matriz, limiar, titulo):
    """
    Desenha a matriz de confusão no limiar escolhido.
    """

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


def _relatar_nivel(classes, probabilidades, medidas, titulo):
    """
    Matriz de confusão, métricas e relatório de classificação de um
    nível (olho ou paciente).
    """

    limiar = medidas["limiar"]

    plotar_matriz_confusao(medidas["matriz_confusao"], limiar, titulo)

    print(f"\n{titulo}:")
    print(f"ROC-AUC: {medidas['roc_auc']:.4f}")
    print(f"PR-AUC: {medidas['pr_auc']:.4f}")
    print(f"Sensibilidade no limiar {limiar:.4f}: {medidas['sensibilidade']:.4f}")
    print(f"Especificidade no limiar {limiar:.4f}: {medidas['especificidade']:.4f}")

    print(
        classification_report(
            classes,
            (probabilidades >= limiar).astype(int),
            target_names=configuracao.ROTULOS_RELATORIO,
            zero_division=0
        )
    )


def mostrar_avaliacao(df, resultado, titulo, conjunto):
    """
    Mostra o que `medir_predicoes` mediu: a comparação dos modos de TTA,
    o kappa da cabeça de grau, os limiares e, por olho e por paciente, a
    matriz de confusão, as métricas e o relatório de classificação.
    """

    print(f"Avaliação principal com {resultado['modo_tta']}.")

    if resultado["limiares_escolhidos"]:
        print(f"\nModos de TTA, {conjunto} por paciente:")
    else:
        print(f"\nModos de TTA, {conjunto} por paciente (limiares da validação):")

    print(resultado["comparacao_tta"].round(4).to_string())

    print(
        f"\nKappa quadrático da cabeça de grau ({conjunto}): "
        f"{resultado['kappa_grau']:.4f} "
        "(Kaggle 2015, topo: 0,85 no teste da competição)"
    )

    if resultado["limiares_escolhidos"]:
        for nivel in ("olho", "paciente"):
            print(f"\nLimiar escolhido por {nivel}: {resultado[f'limiar_{nivel}']:.4f}")

            if nivel in resultado["sem_ponto"]:
                print(
                    "Nenhum limiar atingiu a sensibilidade desejada. "
                    "Foi utilizado o limiar padrão de 0.5."
                )

        print(
            "\nAtenção: os limiares foram escolhidos nesta validação, "
            "então estes números são otimistas. Compare os experimentos por "
            "eles; o teste mede só o modelo final escolhido."
        )

    classes_paciente, predicoes_paciente = agregar_por_paciente(
        df,
        resultado["predicoes"]
    )

    _relatar_nivel(
        df["target"].astype(int).values,
        resultado["predicoes"],
        resultado["metricas_olho"],
        f"{titulo}, por olho"
    )

    _relatar_nivel(
        classes_paciente,
        predicoes_paciente,
        resultado["metricas_paciente"],
        f"{titulo}, por paciente"
    )


# ============================================================
# AVALIAÇÃO COMPLETA
# ============================================================


def _nome_arquivo(texto):
    """
    Converte um título (por exemplo, "Fase 2") em parte de nome de
    arquivo ("fase_2").
    """

    return texto.lower().replace(" ", "_")


def salvar_predicoes(tabela, nome, pasta_resultados=None):
    """
    Grava a tabela de `tabela_predicoes` em
    `pasta_resultados/predicoes_<nome>.csv` (padrão: a pasta da
    execução).
    """

    if pasta_resultados is None:
        pasta_resultados = configuracao.PASTA_RESULTADOS

    os.makedirs(pasta_resultados, exist_ok=True)

    caminho = os.path.join(pasta_resultados, f"predicoes_{nome}.csv")

    tabela.to_csv(caminho, index=False)

    print(f"\nPredições salvas em: {caminho}")

    return caminho


def avaliar_predicoes(
    df,
    variantes,
    nome_arquivo,
    titulo,
    conjunto,
    modo_tta=None,
    limiares=None,
    sensibilidade_desejada=None,
    pasta_resultados=None
):
    """
    A avaliação de um conjunto depois da predição: grava as predições em
    CSV, mede (`medir_predicoes`) e mostra (`mostrar_avaliacao`).
    `variantes` vem de `prever_variantes` ou de qualquer modelo que
    entregue as mesmas predições por modo de TTA.

    Retorna o resultado de `medir_predicoes` com a tabela de predições
    (chave "tabela").
    """

    if modo_tta is None:
        modo_tta = configuracao.MODO_TTA

    tabela = tabela_predicoes(df, variantes, modo_tta)

    salvar_predicoes(tabela, nome_arquivo, pasta_resultados)

    resultado = medir_predicoes(
        df,
        variantes,
        modo_tta,
        limiares,
        sensibilidade_desejada
    )

    resultado["tabela"] = tabela

    mostrar_avaliacao(df, resultado, titulo, conjunto)

    return resultado


def avaliar_modelo(
    modelo,
    val_data,
    df_val,
    historico,
    nome_fase,
    sensibilidade_desejada=None,
    pasta_resultados=None,
    modo_tta=None
):
    """
    Executa a avaliação completa de uma fase de treino na validação:
    gráficos do histórico (se `historico` não for None), escolha dos
    limiares (por olho e por paciente), matrizes de confusão e
    relatórios. As predições são salvas em CSV em `pasta_resultados`.

    As quatro variantes da TTA são previstas uma vez: a avaliação
    principal usa `modo_tta` (sem_tta, tta2 ou tta4), e uma tabela
    compara os três modos por paciente, cada um com o seu limiar. O CSV
    guarda as predições dos três modos.

    Retorna um dicionário com os limiares escolhidos, as métricas, o
    kappa da cabeça de grau, a comparação dos modos de TTA e as
    predições da validação (vetor e tabela por imagem). Os limiares são
    reutilizados por `avaliar_no_teste`.
    """

    if historico is not None:
        plotar_historico(historico, nome_fase)

    print("\nGerando probabilidades na validação...")

    variantes = prever_variantes(modelo, val_data)

    return avaliar_predicoes(
        df_val,
        variantes,
        f"validacao_{_nome_arquivo(nome_fase)}",
        f"{nome_fase} — validação",
        "validação",
        modo_tta=modo_tta,
        sensibilidade_desejada=sensibilidade_desejada,
        pasta_resultados=pasta_resultados
    )


def avaliar_no_teste(
    modelo,
    teste_data,
    df_teste,
    resultado_validacao,
    nome="Teste",
    pasta_resultados=None,
    liberado=None
):
    """
    Avalia o modelo no conjunto de teste usando os limiares escolhidos
    na validação (o resultado de `avaliar_modelo`). É a estimativa
    honesta do desempenho do modelo. As predições são salvas em CSV em
    `pasta_resultados`.

    Só roda com `liberado=True` (padrão: RETINOPATIA_AVALIAR_TESTE=1,
    modelo final). Nos experimentos devolve None sem tocar no teste.

    A avaliação principal usa o modo de TTA da validação. Os três modos
    também são medidos por paciente, cada um com o limiar que a
    validação escolheu para ele: o teste diz quanto custa, em acerto,
    rodar a rede 1, 2 ou 4 vezes no celular.

    Retorna o resultado de `medir_predicoes` com a tabela de predições.
    """

    if liberado is None:
        liberado = configuracao.AVALIAR_TESTE

    if not liberado:
        print(
            "Teste fechado: os experimentos são comparados na validação. "
            "Para o modelo final, RETINOPATIA_AVALIAR_TESTE=1 na primeira "
            "célula."
        )
        return None

    print("\nGerando probabilidades no teste...")

    variantes = prever_variantes(modelo, teste_data)

    return avaliar_predicoes(
        df_teste,
        variantes,
        _nome_arquivo(nome),
        nome,
        "teste",
        modo_tta=resultado_validacao.get("modo_tta", configuracao.MODO_TTA),
        limiares=limiares_da_validacao(resultado_validacao),
        pasta_resultados=pasta_resultados
    )


# ============================================================
# REGISTRO DOS EXPERIMENTOS
# ============================================================


def identificacao_execucao():
    """
    As colunas que identificam a execução no resumo: experimento,
    commit e a configuração principal, lidas da `configuracao` ativa.
    """

    return {
        "experimento": configuracao.NOME_EXPERIMENTO,
        "nome_experimento": configuracao.EXPERIMENTO,
        "modelo": configuracao.NOME_MODELO,
        "commit": configuracao.COMMIT_CODIGO,
        "backbone": configuracao.BACKBONE,
        "tamanho_imagem": configuracao.TAMANHO_IMAGEM[0],
        "pre_processamento": configuracao.VERSAO_PRE_PROCESSAMENTO,
        "tamanho_lote": configuracao.TAMANHO_LOTE,
        "seed_treino": configuracao.SEED_TREINO,
        "modelo_pares": configuracao.MODELO_PARES,
        "destilacao": configuracao.DESTILACAO,
        "fumaca": configuracao.FUMACA
    }


def registrar_experimento(
    resultado_validacao,
    resultado_teste=None,
    pasta_resultados=None,
    extras=None,
    identificacao=None
):
    """
    Grava o resumo desta execução em `pasta_resultados/resumo.csv`:
    a identificação da execução (padrão: `identificacao_execucao`),
    limiares, métricas de validação por olho e por paciente, kappa da
    cabeça de grau e a especificidade de cada modo de TTA. As métricas
    de teste só entram quando `resultado_teste` é passado (modelo final).

    A tabela geral é montada depois por `consolidar_experimentos`: cada
    sessão grava só o próprio resumo, e sessões em paralelo não perdem
    linhas.

    Para escolher entre experimentos, compare as colunas de validação;
    as de teste só valem para o modelo final escolhido. `extras` acrescenta
    colunas (por exemplo, o arquivo do modelo avaliado).
    """

    if pasta_resultados is None:
        pasta_resultados = configuracao.PASTA_RESULTADOS

    if identificacao is None:
        identificacao = identificacao_execucao()

    linha = {
        "data": pd.Timestamp.now().isoformat(timespec="seconds"),
        **identificacao,
        "modo_tta": resultado_validacao.get("modo_tta"),
        "limiar_olho": resultado_validacao["limiar_olho"],
        "limiar_paciente": resultado_validacao["limiar_paciente"],
        "val_kappa_grau": round(float(resultado_validacao["kappa_grau"]), 4)
    }

    for conjunto, resultado in (
        ("val", resultado_validacao),
        ("teste", resultado_teste)
    ):
        if resultado is None:
            continue

        for nivel in ("olho", "paciente"):
            medidas = resultado[f"metricas_{nivel}"]

            for nome in ("roc_auc", "pr_auc", "sensibilidade", "especificidade"):
                linha[f"{conjunto}_{nivel}_{nome}"] = round(
                    float(medidas[nome]),
                    4
                )

    for conjunto, resultado in (
        ("val", resultado_validacao),
        ("teste", resultado_teste)
    ):
        if resultado is None or "comparacao_tta" not in resultado:
            continue

        for modo, medidas in resultado["comparacao_tta"].iterrows():
            for nome in ("roc_auc", "especificidade"):
                valor = medidas[nome]
                linha[f"{conjunto}_paciente_{nome}_{modo}"] = (
                    None if pd.isna(valor) else round(float(valor), 4)
                )

    linha.update(extras or {})

    os.makedirs(
        pasta_resultados,
        exist_ok=True
    )

    caminho = os.path.join(pasta_resultados, "resumo.csv")

    resumo = pd.DataFrame([linha])
    resumo.to_csv(caminho, index=False)

    print(f"Resumo do experimento salvo em: {caminho}")

    return resumo


def consolidar_experimentos(
    pasta_resultados_geral=None,
    caminho_registro=None
):
    """
    Atualiza o registro geral de experimentos com os `resumo.csv` de
    todas as pastas de resultados. Chame em uma sessão só, depois que os
    treinos paralelos terminarem.

    As linhas antigas do registro (treinos de antes dos resumos por
    execução) são preservadas; uma execução que aparece de novo fica com
    a versão do seu resumo.

    Retorna o registro consolidado.
    """

    if pasta_resultados_geral is None:
        pasta_resultados_geral = configuracao.PASTA_RESULTADOS_GERAL

    if caminho_registro is None:
        caminho_registro = configuracao.CAMINHO_REGISTRO_EXPERIMENTOS

    tabelas = []

    if os.path.isfile(caminho_registro):
        tabelas.append(pd.read_csv(caminho_registro))

    for nome in sorted(os.listdir(pasta_resultados_geral)):
        caminho_resumo = os.path.join(
            pasta_resultados_geral,
            nome,
            "resumo.csv"
        )

        if os.path.isfile(caminho_resumo):
            tabelas.append(pd.read_csv(caminho_resumo))

    if not tabelas:
        print("Nenhum resumo encontrado.")
        return pd.DataFrame()

    registro = pd.concat(tabelas, ignore_index=True).drop_duplicates(
        subset="experimento",
        keep="last"
    )

    # Execuções de fumaça só conferem o código; não entram no registro
    if "fumaca" in registro:
        registro = registro[
            ~registro["fumaca"].astype(str).str.lower().eq("true")
        ]

    temporario = caminho_registro + ".parcial"
    registro.to_csv(temporario, index=False)
    os.replace(temporario, caminho_registro)

    print(
        f"Registro consolidado ({len(registro)} execuções): "
        f"{caminho_registro}"
    )

    return registro
