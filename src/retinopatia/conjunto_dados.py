"""
Carregamento do CSV, engenharia dos rótulos e divisão estratificada
agrupada por paciente.

O grau original de severidade (0 a 4) é preservado na coluna
`level_original` e colapsado em uma classe binária na coluna `target`:
grau 0 vira 0 (saudável) e os graus 1 a 4 viram 1 (com retinopatia).
"""

import os

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


def dividir_treino_validacao(
    dados,
    seed=configuracao.SEED,
    numero_splits=configuracao.NUMERO_SPLITS
):
    """
    Divide os dados em treino e validação mantendo os dois olhos do
    mesmo paciente no mesmo conjunto e estratificando pelos cinco graus
    originais de severidade.
    """

    divisor = StratifiedGroupKFold(
        n_splits=numero_splits,
        shuffle=True,
        random_state=seed
    )

    indices_treino, indices_val = next(
        divisor.split(
            X=dados["image"],

            # Estratifica usando os cinco graus originais
            y=dados["level_original"],

            # Mantém os dois olhos do mesmo paciente juntos
            groups=dados["patient_id"]
        )
    )

    df_treino = dados.iloc[
        indices_treino
    ].copy()

    df_val = dados.iloc[
        indices_val
    ].copy()

    # Verifica se há pacientes repetidos entre os conjuntos
    pacientes_treino = set(
        df_treino["patient_id"]
    )

    pacientes_val = set(
        df_val["patient_id"]
    )

    assert pacientes_treino.isdisjoint(
        pacientes_val
    ), (
        "Erro: existem pacientes repetidos "
        "entre treino e validação."
    )

    print(
        "\nNenhum paciente aparece simultaneamente "
        "no treino e na validação."
    )

    # O flow_from_dataframe com class_mode="binary"
    # trabalha corretamente com as classes como strings.
    df_treino["target"] = (
        df_treino["target"]
        .astype(str)
    )

    df_val["target"] = (
        df_val["target"]
        .astype(str)
    )

    return df_treino, df_val


def salvar_splits(
    df_treino,
    df_val,
    pasta_splits=configuracao.PASTA_SPLITS,
    seed=configuracao.SEED
):
    """
    Salva os conjuntos no Google Drive em arquivos versionados pela
    semente, permitindo reproduzir a mesma divisão em outras sessões.
    """

    # Cria a pasta caso ainda não exista
    os.makedirs(
        pasta_splits,
        exist_ok=True
    )

    caminho_split_treino = os.path.join(
        pasta_splits,
        f"treino_seed_{seed}.csv"
    )

    caminho_split_validacao = os.path.join(
        pasta_splits,
        f"validacao_seed_{seed}.csv"
    )

    df_treino.to_csv(
        caminho_split_treino,
        index=False
    )

    df_val.to_csv(
        caminho_split_validacao,
        index=False
    )

    print("\nSplits salvos com sucesso:")

    print(
        "Treino:",
        caminho_split_treino
    )

    print(
        "Validação:",
        caminho_split_validacao
    )

    return caminho_split_treino, caminho_split_validacao


def resumir_conjuntos(dados, df_treino, df_val):
    """
    Mostra o tamanho dos conjuntos e a distribuição das classes.
    """

    print(
        f"\nTotal de imagens originais: "
        f"{len(dados)}"
    )

    print(
        f"Imagens separadas para TREINO: "
        f"{len(df_treino)}"
    )

    print(
        f"Imagens separadas para VALIDAÇÃO: "
        f"{len(df_val)}"
    )

    print("\nDistribuição dos graus originais no TREINO:")
    _mostrar_distribuicao(df_treino, "level_original")

    print("\nDistribuição dos graus originais na VALIDAÇÃO:")
    _mostrar_distribuicao(df_val, "level_original")

    print("\nDistribuição binária no TREINO:")
    _mostrar_distribuicao(df_treino, "target")

    print("\nDistribuição binária na VALIDAÇÃO:")
    _mostrar_distribuicao(df_val, "target")


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
    classe binária, identificação do paciente, divisão agrupada e
    salvamento dos splits.

    Retorna:
        (dados, df_treino, df_val)
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

    df_treino, df_val = dividir_treino_validacao(
        dados,
        seed=seed
    )

    salvar_splits(
        df_treino,
        df_val,
        pasta_splits=pasta_splits,
        seed=seed
    )

    resumir_conjuntos(
        dados,
        df_treino,
        df_val
    )

    return dados, df_treino, df_val
