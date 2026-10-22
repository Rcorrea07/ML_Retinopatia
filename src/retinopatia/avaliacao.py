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

As predições de cada imagem (probabilidade e grau previsto) são
salvas em CSV na pasta de resultados, para que o diagnóstico possa ser
refeito sem rodar o modelo de novo.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    cohen_kappa_score,
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


# Variantes da test-time augmentation, na ordem em que são previstas.
# "tta2" é a média das duas primeiras (original e espelho horizontal) e
# "tta4" a média das quatro. No celular, cada variante é uma passagem a
# mais pela rede.
_VARIANTES_TTA = [
    lambda x: x,
    tf.image.flip_left_right,
    tf.image.flip_up_down,
    lambda x: tf.image.flip_up_down(tf.image.flip_left_right(x))
]

MODOS_TTA = {
    "sem_tta": 1,
    "tta2": 2,
    "tta4": 4
}


def _variantes_tta(tta):
    """
    Transformações aplicadas às imagens na test-time augmentation.
    """

    return _VARIANTES_TTA if tta else _VARIANTES_TTA[:1]


def _prever_variante(modelo, dados, transformar):
    """
    Probabilidades e graus de uma variante, na ordem do pipeline (que
    não embaralha). No modelo de pares a entrada é um dicionário com os
    dois olhos, e a transformação vale para os dois.
    """

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
        for transformar in _VARIANTES_TTA
    ]

    return {
        modo: (
            np.mean([v[0] for v in variantes[:quantidade]], axis=0),
            np.mean([v[1] for v in variantes[:quantidade]], axis=0)
        )
        for modo, quantidade in MODOS_TTA.items()
    }


def prever_saidas(
    modelo,
    dados,
    tta=configuracao.TTA_ATIVO
):
    """
    Gera, para cada imagem e na ordem do pipeline (que não embaralha),
    a probabilidade de retinopatia e o grau previsto pela cabeça
    auxiliar. Com TTA, devolve a média das predições da imagem original
    e de seus espelhamentos.

    Retorna:
        (probabilidades, graus_previstos)
    """

    variantes = [
        _prever_variante(modelo, dados, transformar)
        for transformar in _variantes_tta(tta)
    ]

    return (
        np.mean([v[0] for v in variantes], axis=0),
        np.mean([v[1] for v in variantes], axis=0)
    )


def kappa_grau(niveis_reais, graus_previstos):
    """
    Kappa quadrático da cabeça de grau (regressão arredondada para o
    inteiro mais próximo, limitada a 0–4). É a métrica do Kaggle EyePACS
    2015 (topo: 0,85 no teste da competição), calculada aqui na
    validação do projeto: serve de referência, não de comparação exata.
    """

    previstos = np.clip(np.rint(graus_previstos), 0, 4).astype(int)

    return cohen_kappa_score(
        np.asarray(niveis_reais).astype(int),
        previstos,
        weights="quadratic"
    )


def _agregar(valores, funcao):
    """
    Combina os valores dos olhos de um paciente:
    "max" (o maior), "media" ou "max_media" (média dos dois anteriores).
    """

    if funcao == "max":
        return valores.max()

    if funcao == "media":
        return valores.mean()

    if funcao == "max_media":
        return (valores.max() + valores.mean()) / 2

    raise ValueError(f"Agregação desconhecida: {funcao}")


def agregar_por_paciente(df, probabilidades, funcao="max"):
    """
    Combina os dois olhos de cada paciente: a classe do paciente é 1 se
    qualquer olho estiver doente, e a probabilidade é, por padrão, a
    maior entre as dos dois olhos (ver `_agregar` para as alternativas).

    Retorna:
        (classes_paciente, probabilidades_paciente)
    """

    por_olho = pd.DataFrame({
        "patient_id": df["patient_id"].values,
        "target": df["target"].astype(int).values,
        "probabilidade": probabilidades
    })

    por_paciente = por_olho.groupby("patient_id").agg(
        target=("target", "max"),
        probabilidade=(
            "probabilidade",
            lambda valores: _agregar(valores, funcao)
        )
    )

    return (
        por_paciente["target"].values,
        por_paciente["probabilidade"].values
    )


def salvar_predicoes(
    df,
    probabilidades,
    graus_previstos,
    nome,
    pasta_resultados=configuracao.PASTA_RESULTADOS,
    colunas_extras=None
):
    """
    Salva a tabela de predições por imagem em
    `pasta_resultados/predicoes_<nome>.csv` e a devolve.

    Colunas: image, patient_id, level_original, target,
    prob_doente, grau_previsto e as de `colunas_extras` (por exemplo, as
    predições de cada modo de TTA).
    """

    tabela = df[
        ["image", "patient_id", "level_original", "target"]
    ].copy()

    tabela["prob_doente"] = probabilidades
    tabela["grau_previsto"] = graus_previstos

    for coluna, valores in (colunas_extras or {}).items():
        tabela[coluna] = valores

    os.makedirs(
        pasta_resultados,
        exist_ok=True
    )

    caminho = os.path.join(
        pasta_resultados,
        f"predicoes_{nome}.csv"
    )

    tabela.to_csv(caminho, index=False)

    print(f"\nPredições salvas em: {caminho}")

    return tabela


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


def _nome_arquivo(texto):
    """
    Converte um título (por exemplo, "Fase 2") em parte de nome de
    arquivo ("fase_2").
    """

    return texto.lower().replace(" ", "_")


def avaliar_modelo(
    modelo,
    val_data,
    df_val,
    historico,
    nome_fase,
    sensibilidade_desejada=configuracao.SENSIBILIDADE_DESEJADA,
    pasta_resultados=configuracao.PASTA_RESULTADOS,
    tta=configuracao.TTA_ATIVO
):
    """
    Executa a avaliação completa de uma fase de treino na validação:
    gráficos do histórico (se `historico` não for None), escolha dos
    limiares (por olho e por paciente), matrizes de confusão e
    relatórios. As predições são salvas em CSV em `pasta_resultados`.

    As quatro variantes da TTA são previstas uma vez: a avaliação
    principal usa TTA de 4 (ou nenhuma, com `tta=False`), e uma tabela
    compara sem TTA, TTA de 2 e TTA de 4 por paciente. O CSV guarda as
    predições dos três modos.

    Retorna um dicionário com os limiares escolhidos, as métricas, o
    kappa da cabeça de grau, a comparação dos modos de TTA e as
    predições da validação (vetor e tabela por imagem). Os limiares são
    reutilizados por `avaliar_no_teste`.
    """

    # 1. GRÁFICOS DO TREINAMENTO
    if historico is not None:
        plotar_historico(historico, nome_fase)

    # 2. PROBABILIDADES NA VALIDAÇÃO
    print("\nGerando probabilidades na validação...")

    variantes = prever_variantes(modelo, val_data)

    predicoes, graus_previstos = variantes["tta4" if tta else "sem_tta"]

    colunas_extras = {}

    for modo, (probabilidades_modo, graus_modo) in variantes.items():
        colunas_extras[f"prob_doente_{modo}"] = probabilidades_modo
        colunas_extras[f"grau_previsto_{modo}"] = graus_modo

    tabela = salvar_predicoes(
        df_val,
        predicoes,
        graus_previstos,
        f"validacao_{_nome_arquivo(nome_fase)}",
        pasta_resultados,
        colunas_extras
    )

    comparacao_tta = comparar_modos_tta(
        df_val,
        variantes,
        sensibilidade_desejada
    )

    kappa = kappa_grau(df_val["level_original"], graus_previstos)

    print(
        f"\nKappa quadrático da cabeça de grau (validação): {kappa:.4f} "
        "(Kaggle 2015, topo: 0,85 no teste da competição)"
    )

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
        "então estes números são otimistas. Compare os experimentos por "
        "eles; o teste mede só o modelo final escolhido."
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
        "kappa_grau": kappa,
        "comparacao_tta": comparacao_tta,
        "predicoes": predicoes,
        "tabela": tabela
    }


def comparar_modos_tta(
    df,
    variantes,
    sensibilidade_desejada=configuracao.SENSIBILIDADE_DESEJADA
):
    """
    Compara, por paciente, as predições sem TTA, com TTA de 2 e de 4
    variantes: ROC-AUC, AP e especificidade no limiar que dá a
    sensibilidade desejada (escolhido em cada modo). Um limiar escolhido
    com TTA não vale para a inferência sem TTA, e vice-versa.

    Retorna um DataFrame com uma linha por modo.
    """

    linhas = []

    for modo, (probabilidades, _) in variantes.items():
        classes, probabilidades_paciente = agregar_por_paciente(
            df,
            probabilidades
        )

        limiar, sensibilidade, especificidade = escolher_limiar(
            classes,
            probabilidades_paciente,
            sensibilidade_desejada
        )

        linhas.append({
            "modo": modo,
            "passagens": MODOS_TTA[modo],
            "roc_auc": roc_auc_score(classes, probabilidades_paciente),
            "pr_auc": average_precision_score(
                classes,
                probabilidades_paciente
            ),
            "sensibilidade": sensibilidade,
            "especificidade": especificidade,
            "limiar": limiar
        })

    tabela = pd.DataFrame(linhas).set_index("modo")

    print("\nModos de TTA, validação por paciente:")
    print(tabela.round(4).to_string())

    return tabela


def avaliar_no_teste(
    modelo,
    teste_data,
    df_teste,
    resultado_validacao,
    nome="Teste",
    pasta_resultados=configuracao.PASTA_RESULTADOS,
    liberado=configuracao.AVALIAR_TESTE
):
    """
    Avalia o modelo no conjunto de teste usando os limiares escolhidos
    na validação (o resultado de `avaliar_modelo`). É a estimativa
    honesta do desempenho do modelo. As predições são salvas em CSV em
    `pasta_resultados`.

    Só roda com `liberado=True` (RETINOPATIA_AVALIAR_TESTE=1, modelo
    final). Nos experimentos devolve None sem tocar no teste.

    Retorna um dicionário com as métricas por olho e por paciente e as
    predições do teste (vetor e tabela por imagem).
    """

    if not liberado:
        print(
            "Teste fechado: os experimentos são comparados na validação. "
            "Para o modelo final, RETINOPATIA_AVALIAR_TESTE=1 na primeira "
            "célula."
        )
        return None

    print("\nGerando probabilidades no teste...")

    predicoes, graus_previstos = prever_saidas(modelo, teste_data)

    tabela = salvar_predicoes(
        df_teste,
        predicoes,
        graus_previstos,
        _nome_arquivo(nome),
        pasta_resultados
    )

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
        "predicoes": predicoes,
        "tabela": tabela
    }


def registrar_experimento(
    resultado_validacao,
    resultado_teste=None,
    pasta_resultados=configuracao.PASTA_RESULTADOS
):
    """
    Grava o resumo desta execução em `pasta_resultados/resumo.csv`:
    experimento, commit, configuração principal, limiares, métricas de
    validação por olho e por paciente, kappa da cabeça de grau e a
    especificidade de cada modo de TTA. As métricas de teste só entram
    quando `resultado_teste` é passado (modelo final).

    A tabela geral é montada depois por `consolidar_experimentos`: cada
    sessão grava só o próprio resumo, e sessões em paralelo não perdem
    linhas.

    Para escolher entre experimentos, compare as colunas de validação;
    as de teste só valem para o modelo final escolhido.
    """

    linha = {
        "data": pd.Timestamp.now().isoformat(timespec="seconds"),
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
        "fumaca": configuracao.FUMACA,
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
            metricas = resultado[f"metricas_{nivel}"]

            for nome in ("roc_auc", "pr_auc", "sensibilidade", "especificidade"):
                linha[f"{conjunto}_{nivel}_{nome}"] = round(
                    float(metricas[nome]),
                    4
                )

    for modo, metricas in resultado_validacao["comparacao_tta"].iterrows():
        for nome in ("roc_auc", "especificidade"):
            valor = metricas[nome]
            linha[f"val_paciente_{nome}_{modo}"] = (
                None if pd.isna(valor) else round(float(valor), 4)
            )

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
    pasta_resultados_geral=configuracao.PASTA_RESULTADOS_GERAL,
    caminho_registro=configuracao.CAMINHO_REGISTRO_EXPERIMENTOS
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
