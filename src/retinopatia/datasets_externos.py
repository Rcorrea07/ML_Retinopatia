"""
Datasets de retinopatia além do EyePACS, só para o treino.

DDR, APTOS 2019, DeepDRiD (fotos comuns, sem as de campo ultra-amplo) e
IDRiD (parte B, de graduação) ficam na pasta Datasets do Drive, cada um
no zip em que foi baixado (notebook Baixar_Datasets; motivos e licenças
em docs/DATASETS.md). Todos usam a escala ICDR de 0 a 4, a mesma do
EyePACS.

Para cada dataset, um leitor monta a tabela das fotos direto do zip, sem
extraí-lo: o caminho da foto dentro do zip (`membro`), o nome da imagem
processada (`image`, com o prefixo do dataset e sempre .jpg), o grau
(`level`) e o paciente (`patient_id`, com o mesmo prefixo; quando o
dataset não informa o paciente, cada foto é o seu). Daí em diante as
fotos seguem o caminho das extras de 2015: o mesmo pré-processamento,
uma pasta e um cache por dataset (`ambiente_colab.preparar_imagens_externas`)
e linhas acrescentadas ao treino com `pasta_processada` e `origem`.

Os caminhos dentro dos zips foram conferidos nos arquivos baixados em
30/10/2026. Este módulo usa só pandas (e openpyxl para a planilha do
DeepDRiD): os leitores podem ser testados sem TensorFlow.
"""

import io
import os
import posixpath
import re
import zipfile

import pandas as pd

from . import configuracao

EXTENSOES_FOTOS = (".jpg", ".jpeg", ".png", ".tif")


# ============================================================
# AJUDANTES
# ============================================================

def _membro(nomes, final):
    """
    O único arquivo do zip cujo caminho termina em `final`.
    """

    encontrados = [nome for nome in nomes if nome.endswith(final)]

    if len(encontrados) != 1:
        raise ValueError(
            f"Esperado um arquivo terminado em '{final}' no zip; "
            f"encontrados: {encontrados[:5]}"
        )

    return encontrados[0]


def _fotos(nomes, trecho):
    """
    Fotos do zip cujo caminho contém `trecho`, pelo nome do arquivo sem
    extensão. Dois arquivos com o mesmo nome ali tornariam a ligação com
    a tabela de rótulos ambígua: a execução para.
    """

    fotos = {}

    for nome in nomes:
        if trecho not in nome or not nome.lower().endswith(EXTENSOES_FOTOS):
            continue

        chave = posixpath.splitext(posixpath.basename(nome))[0]

        if chave in fotos:
            raise ValueError(
                f"Duas fotos chamadas '{chave}' em '{trecho}': "
                f"{fotos[chave]} e {nome}"
            )

        fotos[chave] = nome

    return fotos


def _ler_tabela(arquivo_zip, membro):
    """
    Lê um CSV ou uma planilha de dentro do zip.
    """

    conteudo = io.BytesIO(arquivo_zip.read(membro))

    if membro.lower().endswith(".csv"):
        tabela = pd.read_csv(conteudo)
    else:
        tabela = pd.read_excel(conteudo)

    tabela.columns = [str(coluna).strip() for coluna in tabela.columns]

    return tabela


def _montar(fotos, chaves, niveis, prefixo, pacientes=None):
    """
    Tabela de um dataset no formato comum. Uma chave da tabela de
    rótulos sem foto no zip para a execução.
    """

    chaves = [str(chave).strip() for chave in chaves]

    sem_foto = [chave for chave in chaves if chave not in fotos]

    if sem_foto:
        raise ValueError(
            f"{len(sem_foto)} rótulos de {prefixo} sem foto no zip, "
            f"por exemplo {sem_foto[:5]}"
        )

    if pacientes is None:
        pacientes = chaves

    return pd.DataFrame({
        "membro": [fotos[chave] for chave in chaves],
        "image": [f"{prefixo}_{chave}.jpg" for chave in chaves],
        "level": list(niveis),
        "patient_id": [f"{prefixo}_{paciente}" for paciente in pacientes],
    })


# ============================================================
# LEITORES (um por dataset)
# ============================================================

def tabela_ddr(arquivo_zip):
    """
    DDR, cópia do Kaggle (mariaherrerot/ddrdataset): 12.522 fotos
    graduáveis em DR_grading/DR_grading/, rótulos em DR_grading.csv
    (id_code com a extensão, diagnosis). Sem identificação do paciente.
    """

    nomes = arquivo_zip.namelist()

    rotulos = _ler_tabela(arquivo_zip, _membro(nomes, "DR_grading.csv"))

    return _montar(
        _fotos(nomes, "DR_grading/"),
        rotulos["id_code"].map(lambda nome: posixpath.splitext(nome)[0]),
        rotulos["diagnosis"],
        "ddr"
    )


def tabela_aptos(arquivo_zip):
    """
    APTOS 2019 (competição do Kaggle): 3.662 fotos com rótulo em
    train_images/ (png), rótulos em train.csv (id_code, diagnosis). O
    test da competição não tem rótulo e fica de fora.
    """

    nomes = arquivo_zip.namelist()

    rotulos = _ler_tabela(arquivo_zip, _membro(nomes, "train.csv"))

    return _montar(
        _fotos(nomes, "train_images/"),
        rotulos["id_code"],
        rotulos["diagnosis"],
        "aptos"
    )


def tabela_deepdrid(
    arquivo_zip,
    qualidade_minima=configuracao.QUALIDADE_MINIMA_DEEPDRID
):
    """
    DeepDRiD (GitHub deepdrdoc), só as fotos comuns de fundo de olho
    (regular_fundus_images; as de campo ultra-amplo são outra
    modalidade): 2.000 fotos de 500 pacientes, duas por olho, nos
    conjuntos de treino, validação e avaliação do desafio.

    Treino e validação trazem num CSV a nota de qualidade e o grau de
    cada olho (left/right_eye_DR_Level; o olho vem do nome, 1_l1 ou
    1_r2). A avaliação traz o grau numa planilha (Challenge1_labels) e a
    qualidade em outra (Challenge2_labels). Ficam só as fotos com nota de
    qualidade de pelo menos `qualidade_minima` (ver configuracao).
    """

    nomes = arquivo_zip.namelist()
    raiz = "regular_fundus_images/"
    partes = []

    for conjunto, pasta in (
        ("treino", "regular-fundus-training"),
        ("validacao", "regular-fundus-validation"),
    ):
        rotulos = _ler_tabela(
            arquivo_zip,
            _membro(nomes, f"{raiz}{pasta}/{pasta}.csv")
        )

        olho = rotulos["image_id"].astype(str).str.extract(
            r"_([lr])\d+$",
            expand=False
        )

        if olho.isna().any():
            raise ValueError(
                f"Fotos do DeepDRiD ({pasta}) sem o olho no nome: "
                f"{rotulos.loc[olho.isna(), 'image_id'].head().tolist()}"
            )

        rotulos["level"] = rotulos["left_eye_DR_Level"].where(
            olho == "l",
            rotulos["right_eye_DR_Level"]
        )
        rotulos["qualidade"] = rotulos["Overall quality"]

        partes.append((conjunto, f"{raiz}{pasta}/", rotulos))

    pasta = "Online-Challenge1&2-Evaluation"

    graus = _ler_tabela(
        arquivo_zip,
        _membro(nomes, f"{raiz}{pasta}/Challenge1_labels.xlsx")
    )
    qualidade = _ler_tabela(
        arquivo_zip,
        _membro(nomes, f"{raiz}{pasta}/Challenge2_labels.xlsx")
    )

    avaliacao = graus.merge(
        qualidade[["image_id", "Overall quality"]],
        on="image_id",
        how="left",
        validate="one_to_one"
    ).rename(columns={"DR_Levels": "level", "Overall quality": "qualidade"})

    partes.append(("avaliacao", f"{raiz}{pasta}/", avaliacao))

    tabelas = []

    for conjunto, trecho, rotulos in partes:
        aceitas = rotulos[rotulos["qualidade"] >= qualidade_minima]

        print(
            f"DeepDRiD {conjunto}: {len(aceitas)} de {len(rotulos)} fotos "
            "com qualidade aceita"
        )

        # O paciente é o número antes do "_" (1_l1 → 1); conjuntos
        # diferentes do desafio podem repetir números
        chaves = aceitas["image_id"].astype(str)

        tabelas.append(_montar(
            _fotos(nomes, trecho),
            chaves,
            aceitas["level"],
            f"deepdrid_{conjunto}",
            pacientes=chaves.str.split("_").str[0]
        ))

    return pd.concat(tabelas, ignore_index=True)


def tabela_idrid(arquivo_zip):
    """
    IDRiD (cópia do Kaggle aaryapatel98), só a parte B, de graduação:
    413 fotos de treino e 103 de teste do desafio, com rótulos nos CSVs
    de "2. Groundtruths" ('Image name', 'Retinopathy grade'). Os dois
    conjuntos repetem os nomes (IDRiD_001 existe nos dois, fotos
    diferentes), por isso o conjunto entra no nome. As partes A e C
    repetem fotos da B e ficam de fora. Sem identificação do paciente.
    """

    nomes = arquivo_zip.namelist()
    raiz = "B. Disease Grading/"
    tabelas = []

    for conjunto, letra, rotulo_csv in (
        ("training", "a", "IDRiD_Disease Grading_Training Labels.csv"),
        ("testing", "b", "IDRiD_Disease Grading_Testing Labels.csv"),
    ):
        rotulos = _ler_tabela(
            arquivo_zip,
            _membro(nomes, rotulo_csv)
        ).dropna(subset=["Image name"])

        tabelas.append(_montar(
            _fotos(nomes, f"{raiz}1. Original Images/{letra}. "),
            rotulos["Image name"],
            rotulos["Retinopathy grade"],
            f"idrid_{conjunto}"
        ))

    return pd.concat(tabelas, ignore_index=True)


LEITORES = {
    "DDR": tabela_ddr,
    "APTOS": tabela_aptos,
    "DeepDRiD": tabela_deepdrid,
    "IDRiD": tabela_idrid,
}


# ============================================================
# TABELA PARA O TREINO
# ============================================================

def carregar_dataset_externo(nome, caminho_zip=None):
    """
    Tabela das fotos de um dataset externo, lida do zip no Drive, no
    formato do treino: `image`, `membro`, `level_original`, `target`,
    `patient_id`, `pasta_processada` e `origem` (o nome do dataset).
    Graus fora de 0 a 4 (a classe "não graduável" de alguns datasets)
    ficam de fora.
    """

    from . import conjunto_dados

    caminho_zip = caminho_zip or configuracao.ZIPS_DATASETS_EXTERNOS[nome]

    if not os.path.isfile(caminho_zip):
        raise FileNotFoundError(
            f"Zip do {nome} não encontrado em {caminho_zip}. Baixe com o "
            "notebook Baixar_Datasets (conta pessoal) e confira se a pasta "
            "Datasets está compartilhada com a conta do projeto."
        )

    with zipfile.ZipFile(caminho_zip) as arquivo_zip:
        tabela = LEITORES[nome](arquivo_zip)

    niveis = pd.to_numeric(tabela["level"], errors="coerce")
    validos = niveis.isin(configuracao.NIVEIS_PERMITIDOS)

    if not validos.all():
        print(
            f"{nome}: {(~validos).sum()} fotos sem grau de 0 a 4 ficam de "
            f"fora ({niveis[~validos].value_counts(dropna=False).to_dict()})"
        )

    tabela = tabela[validos].reset_index(drop=True)
    tabela["level"] = niveis[validos].astype(int).values

    tabela = conjunto_dados.criar_classe_binaria(tabela)

    tabela["pasta_processada"] = configuracao.PASTAS_IMAGENS_EXTERNAS[nome]
    tabela["origem"] = nome

    print(
        f"\n{nome}: {len(tabela)} fotos "
        f"({tabela['patient_id'].nunique()} pacientes), só para o treino"
    )

    print(
        tabela["level_original"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    return tabela[
        [
            "image", "membro", "level_original", "target", "patient_id",
            "pasta_processada", "origem"
        ]
    ]


def manter_processadas(
    tabela,
    pasta_processada,
    fracao_maxima=configuracao.FRACAO_MAXIMA_FALHAS_EXTERNOS
):
    """
    Só as linhas cuja imagem processada está em `pasta_processada`. As
    que faltam falharam no pré-processamento (ficam fora do treino); se
    forem mais que `fracao_maxima`, o cache está incompleto e a execução
    para.
    """

    existe = tabela["image"].map(
        lambda nome: os.path.isfile(os.path.join(pasta_processada, nome))
    )

    faltam = int((~existe).sum())

    if faltam > fracao_maxima * len(tabela):
        raise RuntimeError(
            f"{faltam} de {len(tabela)} imagens processadas faltam em "
            f"{pasta_processada}; o cache está incompleto."
        )

    if faltam:
        print(
            f"{faltam} imagens que falharam no pré-processamento ficam "
            f"fora do treino: {tabela.loc[~existe, 'image'].head().tolist()}"
        )

    return tabela[existe].reset_index(drop=True)
