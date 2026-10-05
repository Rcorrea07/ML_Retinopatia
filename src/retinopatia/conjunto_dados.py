"""
Carregamento do CSV, engenharia dos rótulos e divisão estratificada
agrupada por paciente em treino, validação e teste.

O grau original de severidade (0 a 4) é preservado na coluna
`level_original` e colapsado em uma classe binária na coluna `target`:
grau 0 vira 0 (saudável) e os graus 1 a 4 viram 1 (com retinopatia).
O modelo usa as duas colunas: `target` na decisão binária e
`level_original` na cabeça auxiliar de grau.

Os três conjuntos têm papéis distintos: o treino ajusta os pesos, a
validação escolhe o checkpoint e o limiar de decisão, e o teste só é
usado no fim, para medir o desempenho sem viés de seleção.
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import configuracao


def carregar_csv(caminho_csv=configuracao.CAMINHO_CSV):
    """
    Lê o CSV de rótulos, acrescenta a extensão .jpeg aos nomes das
    imagens e valida o padrão dos nomes.
    """

    dados = pd.read_csv(caminho_csv)

    # Adiciona .jpeg aos nomes das imagens
    dados["image"] = (
        dados["image"]
        .astype(str)
        .apply(lambda nome: f"{nome}.jpeg")
    )

    validar_padrao_nomes(dados)

    return dados


def validar_padrao_nomes(dados):
    """
    Verifica se todos os nomes seguem o formato esperado:
    123_left.jpeg ou 123_right.jpeg.
    """

    padrao_valido = dados["image"].str.match(
        r"^.+_(?:left|right)\.jpeg$"
    )

    if not padrao_valido.all():
        imagens_fora_do_padrao = dados.loc[
            ~padrao_valido,
            "image"
        ]

        print(
            "\nExemplos de imagens fora do padrão esperado:"
        )

        print(
            imagens_fora_do_padrao
            .head(20)
            .to_string(index=False)
        )

        raise ValueError(
            f"{len(imagens_fora_do_padrao)} imagens "
            "não seguem o padrão esperado de nomes."
        )

    print(
        "\nTodos os nomes seguem o padrão "
        "'identificador_left.jpeg' ou "
        "'identificador_right.jpeg'."
    )


def criar_classe_binaria(dados):
    """
    Preserva o grau original em `level_original` e cria a classe
    binária `target` usada pelo modelo.
    """

    # Converte os níveis para números inteiros.
    # errors="raise" faz o código parar se houver algum valor inválido.
    dados["level_original"] = pd.to_numeric(
        dados["level"],
        errors="raise"
    ).astype("int8")

    # Verifica se existem somente os graus esperados
    niveis_encontrados = set(
        dados["level_original"].unique()
    )

    niveis_invalidos = (
        niveis_encontrados - configuracao.NIVEIS_PERMITIDOS
    )

    if niveis_invalidos:
        raise ValueError(
            f"Níveis inválidos encontrados: {niveis_invalidos}"
        )

    # Grau 0 vira classe 0: saudável
    # Graus 1, 2, 3 e 4 viram classe 1: com retinopatia
    dados["target"] = (
        dados["level_original"] > 0
    ).astype("int8")

    return dados


def extrair_identificador_paciente(dados):
    """
    Extrai o identificador do paciente a partir do nome da imagem.

    Exemplo:
        123_left.jpeg  -> paciente 123
        123_right.jpeg -> paciente 123
    """

    dados["patient_id"] = (
        dados["image"]
        .str.replace(
            r"\.jpeg$",
            "",
            regex=True
        )
        .str.replace(
            r"_(left|right)$",
            "",
            regex=True
        )
    )

    return dados


def dividir_treino_validacao_teste(
    dados,
    seed=configuracao.SEED,
    numero_splits=configuracao.NUMERO_SPLITS,
    dobra_teste=configuracao.DOBRA_TESTE,
    dobra_validacao=configuracao.DOBRA_VALIDACAO
):
    """
    Divide os dados em treino, validação e teste mantendo os dois olhos
    do mesmo paciente no mesmo conjunto e estratificando pelos cinco
    graus originais de severidade.

    Uma dobra do StratifiedGroupKFold vira teste, outra vira validação
    e as demais formam o treino.
    """

    divisor = StratifiedGroupKFold(
        n_splits=numero_splits,
        shuffle=True,
        random_state=seed
    )

    dobras = [
        indices_dobra
        for _, indices_dobra in divisor.split(
            X=dados["image"],

            # Estratifica usando os cinco graus originais
            y=dados["level_original"],

            # Mantém os dois olhos do mesmo paciente juntos
            groups=dados["patient_id"]
        )
    ]

    indices_teste = dobras[dobra_teste]
    indices_val = dobras[dobra_validacao]

    indices_treino = np.setdiff1d(
        np.arange(len(dados)),
        np.concatenate([indices_teste, indices_val])
    )

    df_treino = dados.iloc[indices_treino].copy()
    df_val = dados.iloc[indices_val].copy()
    df_teste = dados.iloc[indices_teste].copy()

    verificar_pacientes_disjuntos(df_treino, df_val, df_teste)

    return df_treino, df_val, df_teste


def verificar_pacientes_disjuntos(df_treino, df_val, df_teste):
    """
    Garante que nenhum paciente aparece em mais de um conjunto.
    """

    pacientes_treino = set(df_treino["patient_id"])
    pacientes_val = set(df_val["patient_id"])
    pacientes_teste = set(df_teste["patient_id"])

    assert pacientes_treino.isdisjoint(pacientes_val), (
        "Erro: existem pacientes repetidos entre treino e validação."
    )

    assert pacientes_treino.isdisjoint(pacientes_teste), (
        "Erro: existem pacientes repetidos entre treino e teste."
    )

    assert pacientes_val.isdisjoint(pacientes_teste), (
        "Erro: existem pacientes repetidos entre validação e teste."
    )

    print(
        "\nNenhum paciente aparece em mais de um conjunto "
        "(treino, validação e teste)."
    )


def _caminhos_splits(
    pasta_splits=configuracao.PASTA_SPLITS,
    seed=configuracao.SEED,
    versao=configuracao.VERSAO_SPLIT
):
    """
    Monta os caminhos dos CSVs de split, versionados pelo esquema de
    divisão e pela semente.
    """

    return {
        nome: os.path.join(
            pasta_splits,
            f"{nome}_{versao}_seed_{seed}.csv"
        )
        for nome in ["treino", "validacao", "teste"]
    }


def salvar_splits(
    df_treino,
    df_val,
    df_teste,
    pasta_splits=configuracao.PASTA_SPLITS,
    seed=configuracao.SEED
):
    """
    Salva os conjuntos no Google Drive, permitindo reproduzir a mesma
    divisão em outras sessões.
    """

    # Cria a pasta caso ainda não exista
    os.makedirs(
        pasta_splits,
        exist_ok=True
    )

    caminhos = _caminhos_splits(pasta_splits, seed)

    for nome, df in zip(
        ["treino", "validacao", "teste"],
        [df_treino, df_val, df_teste]
    ):
        df.to_csv(caminhos[nome], index=False)

    print("\nSplits salvos com sucesso:")

    for nome, caminho in caminhos.items():
        print(f"{nome}: {caminho}")

    return caminhos


def carregar_splits(
    dados,
    pasta_splits=configuracao.PASTA_SPLITS,
    seed=configuracao.SEED
):
    """
    Carrega os splits salvos anteriormente, se os três existirem.

    Retorna (df_treino, df_val, df_teste) ou None quando algum arquivo
    estiver faltando. Interrompe com erro se os splits não baterem com
    o CSV atual (imagens desconhecidas ou pacientes repetidos).
    """

    caminhos = _caminhos_splits(pasta_splits, seed)

    if not all(os.path.isfile(c) for c in caminhos.values()):
        return None

    conjuntos = [
        pd.read_csv(
            caminhos[nome],
            dtype={"image": str, "patient_id": str}
        )
        for nome in ["treino", "validacao", "teste"]
    ]

    imagens_salvas = set().union(
        *(set(df["image"]) for df in conjuntos)
    )

    if imagens_salvas != set(dados["image"]):
        raise ValueError(
            "Os splits salvos não correspondem às imagens do CSV atual. "
            "Apague os arquivos em "
            f"{pasta_splits} ou mude VERSAO_SPLIT."
        )

    verificar_pacientes_disjuntos(*conjuntos)

    print("\nSplits carregados de:")

    for nome, caminho in caminhos.items():
        print(f"{nome}: {caminho}")

    return tuple(conjuntos)


def resumir_conjuntos(dados, df_treino, df_val, df_teste):
    """
    Mostra o tamanho dos conjuntos e a distribuição das classes.
    """

    print(
        f"\nTotal de imagens originais: "
        f"{len(dados)}"
    )

    conjuntos = {
        "TREINO": df_treino,
        "VALIDAÇÃO": df_val,
        "TESTE": df_teste
    }

    for nome, df in conjuntos.items():
        print(
            f"Imagens separadas para {nome}: {len(df)} "
            f"({df['patient_id'].nunique()} pacientes)"
        )

    for nome, df in conjuntos.items():
        print(f"\nDistribuição dos graus originais no {nome}:")
        _mostrar_distribuicao(df, "level_original")

    for nome, df in conjuntos.items():
        print(f"\nDistribuição binária no {nome}:")
        _mostrar_distribuicao(df, "target")


def _mostrar_distribuicao(df, coluna):
    """
    Mostra a distribuição percentual de uma coluna.
    """

    print(
        df[coluna]
        .value_counts(normalize=True)
        .sort_index()
        .apply(
            lambda valor: f"{valor * 100:.2f}%"
        )
    )


def preparar_dataset(
    caminho_csv=configuracao.CAMINHO_CSV,
    pasta_splits=configuracao.PASTA_SPLITS,
    seed=configuracao.SEED
):
    """
    Executa o preparo completo dos dados: leitura do CSV, criação da
    classe binária, identificação do paciente e divisão agrupada em
    treino, validação e teste.

    Se os splits desta versão e semente já estiverem salvos no Drive,
    eles são reutilizados em vez de recalculados.

    Retorna:
        (dados, df_treino, df_val, df_teste)
    """

    dados = carregar_csv(caminho_csv)
    dados = criar_classe_binaria(dados)
    dados = extrair_identificador_paciente(dados)

    print("\nExemplos de imagens, pacientes e classificações:")

    print(
        dados[
            [
                "image",
                "patient_id",
                "level_original",
                "target"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    conjuntos = carregar_splits(
        dados,
        pasta_splits=pasta_splits,
        seed=seed
    )

    if conjuntos is None:
        print("\nNenhum split salvo encontrado. Criando a divisão...")

        conjuntos = dividir_treino_validacao_teste(
            dados,
            seed=seed
        )

        salvar_splits(
            *conjuntos,
            pasta_splits=pasta_splits,
            seed=seed
        )

    df_treino, df_val, df_teste = conjuntos

    resumir_conjuntos(
        dados,
        df_treino,
        df_val,
        df_teste
    )

    return dados, df_treino, df_val, df_teste
