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


def prever_todas_saidas(modelo, dados, tta=configuracao.TTA_ATIVO):
    """Média de todas as saídas, na ordem das imagens, incluindo cinco classes."""
    acumuladas = {}
    for transformar in _variantes_tta(tta):
        imagens = dados.map(lambda x, *_, f=transformar: f(x))
        saida = modelo.predict(imagens, verbose=1)
        if not isinstance(saida, dict):
            saida = dict(zip(modelo.output_names, saida))
        for nome, valores in saida.items():
            acumuladas.setdefault(nome, []).append(np.asarray(valores))
    return {nome: np.mean(valores, axis=0) for nome, valores in acumuladas.items()}


def prever_saidas(modelo, dados, tta=configuracao.TTA_ATIVO):
    """Probabilidade binária e grau, preservando a interface legada."""
    saida = prever_todas_saidas(modelo, dados, tta)
    return (saida[configuracao.SAIDA_DOENTE].ravel(),
            saida[configuracao.SAIDA_GRAU].ravel())


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
    pasta_resultados=configuracao.PASTA_RESULTADOS
):
    """
    Salva a tabela de predições por imagem em
    `pasta_resultados/predicoes_<nome>.csv` e a devolve.

    Colunas: image, patient_id, level_original, target,
    prob_doente, grau_previsto.
    """

    tabela = df[
        ["image", "patient_id", "level_original", "target"]
    ].copy()

    tabela["prob_doente"] = probabilidades
    tabela["grau_previsto"] = graus_previstos

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

    Retorna um dicionário com os limiares escolhidos, as métricas e as
    predições da validação (vetor e tabela por imagem). Os limiares são
    reutilizados por `avaliar_no_teste`.
    """

    # 1. GRÁFICOS DO TREINAMENTO
    if historico is not None:
        plotar_historico(historico, nome_fase)

    # 2. PROBABILIDADES NA VALIDAÇÃO
    print("\nGerando probabilidades na validação...")

    todas_saidas = prever_todas_saidas(modelo, val_data, tta=tta)
    predicoes = todas_saidas[configuracao.SAIDA_DOENTE].ravel()
    graus_previstos = todas_saidas[configuracao.SAIDA_GRAU].ravel()

    tabela = salvar_predicoes(
        df_val,
        predicoes,
        graus_previstos,
        f"validacao_{_nome_arquivo(nome_fase)}",
        pasta_resultados
    )

    if configuracao.SAIDA_CLASSES in todas_saidas:
        classes_prob = todas_saidas[configuracao.SAIDA_CLASSES]
        for grau in range(configuracao.NUMERO_GRAUS):
            tabela[f"prob_grau_{grau}"] = classes_prob[:, grau]
        tabela["classe_grau"] = np.argmax(classes_prob, axis=1)
        tabela.to_csv(os.path.join(pasta_resultados,
            f"predicoes_validacao_{_nome_arquivo(nome_fase)}.csv"), index=False)
        relatar_cinco_graus(tabela, nome_fase, pasta_resultados)

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
        "então estes números são otimistas. O teste histórico já foi "
        "consultado; uma nova estimativa independente exige dados externos retidos."
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
        "predicoes": predicoes,
        "tabela": tabela
    }


def avaliar_no_teste(
    modelo,
    teste_data,
    df_teste,
    resultado_validacao,
    nome="Teste",
    pasta_resultados=configuracao.PASTA_RESULTADOS
):
    """
    Avalia o modelo no conjunto de teste usando os limiares escolhidos
    na validação (o resultado de `avaliar_modelo`). É a estimativa
    honesta do desempenho do modelo. As predições são salvas em CSV em
    `pasta_resultados`.

    Retorna um dicionário com as métricas por olho e por paciente e as
    predições do teste (vetor e tabela por imagem).
    """

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
    caminho_registro=None
):
    """
    Grava um resumo individual no Drive: commit do código, configuração,
    limiares e métricas de validação por olho e por paciente. Teste é opcional.
    O CSV global só é atualizado pelo consolidador após as filas.

    Para escolher entre experimentos, compare as colunas de validação;
    as de teste só valem para o modelo final escolhido.
    """

    if caminho_registro is None:
        # Cada processo publica só a própria linha. O consolidador tem escritor único.
        caminho_registro = os.path.join(configuracao.PASTA_RESULTADOS, "resumo.csv")
    linha = {
        "perfil": configuracao.PERFIL_EXPERIMENTO,
        "seed_split": configuracao.SEED_SPLIT,
        "seed_treino": configuracao.SEED_TREINO,
        "hash_configuracao": configuracao.HASH_CONFIGURACAO,
        "ensaio": configuracao.ENSAIO,
        "data": pd.Timestamp.now().isoformat(timespec="seconds"),
        "experimento": configuracao.NOME_EXPERIMENTO,
        "modelo": configuracao.NOME_MODELO,
        "commit": configuracao.COMMIT_CODIGO,
        "backbone": configuracao.BACKBONE,
        "tamanho_imagem": configuracao.TAMANHO_IMAGEM[0],
        "pre_processamento": configuracao.VERSAO_PRE_PROCESSAMENTO,
        "tamanho_lote": configuracao.TAMANHO_LOTE,
        "limiar_olho": resultado_validacao["limiar_olho"],
        "limiar_paciente": resultado_validacao["limiar_paciente"]
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

    os.makedirs(
        os.path.dirname(caminho_registro),
        exist_ok=True
    )

    registro = pd.DataFrame([linha])

    if os.path.isfile(caminho_registro):
        registro = pd.concat(
            [pd.read_csv(caminho_registro), registro],
            ignore_index=True
        )

    registro.to_csv(caminho_registro, index=False)

    print(f"Experimento registrado em: {caminho_registro}")

    return registro


def relatar_cinco_graus(tabela, nome, pasta_resultados):
    """Persistência das cinco classes reais; não existe um sexto rótulo."""
    from sklearn.metrics import cohen_kappa_score
    from .persistencia import salvar_json
    reais = tabela["level_original"].astype(int)
    previstas = tabela["classe_grau"].astype(int)
    matriz = confusion_matrix(reais, previstas, labels=range(5))
    relatorio = classification_report(reais, previstas, labels=range(5),
        target_names=configuracao.NOMES_GRAUS, output_dict=True, zero_division=0)
    kappa = cohen_kappa_score(reais, previstas, weights="quadratic", labels=range(5))
    pd.DataFrame(matriz, index=range(5), columns=range(5)).to_csv(
        os.path.join(pasta_resultados, f"matriz_5_graus_{_nome_arquivo(nome)}.csv"))
    salvar_json(os.path.join(pasta_resultados, f"relatorio_5_graus_{_nome_arquivo(nome)}.json"),
                {"kappa_quadratico": float(kappa), "relatorio": relatorio})
    print(f"Cinco graus — kappa quadrático: {kappa:.4f}")
    return relatorio


def consolidar_experimentos(pasta_resultados=None):
    """Executar em apenas uma sessão após as filas: preserva linhas antigas."""
    from pathlib import Path
    import json
    pasta = Path(pasta_resultados or os.path.dirname(configuracao.PASTA_RESULTADOS))
    destino = pasta / "registro_experimentos.csv"
    tabelas = [pd.read_csv(destino)] if destino.exists() else []
    for resumo in sorted(pasta.glob("*/resumo.csv")):
        estado = resumo.parent / "estado_execucao.json"
        if estado.exists() and json.loads(estado.read_text()).get("estado") != "concluido":
            continue
        tabela = pd.read_csv(resumo)
        if "ensaio" in tabela:
            tabela = tabela[~tabela["ensaio"].astype(str).str.lower().eq("true")]
        tabelas.append(tabela)
    if not tabelas:
        return pd.DataFrame()
    resultado = pd.concat(tabelas, ignore_index=True)
    resultado = resultado.drop_duplicates(subset=["experimento"], keep="last")
    temporario = str(destino) + ".parcial"
    resultado.to_csv(temporario, index=False)
    os.replace(temporario, destino)
    return resultado
