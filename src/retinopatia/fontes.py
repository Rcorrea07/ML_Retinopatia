"""
Fontes de imagens além do train do EyePACS, numa ficha por fonte, e um
preparo só para todas.

Cada fonte (`FonteDados`) diz de onde vêm as fotos, como montar a tabela
de rótulos, como extrair as fotos originais, onde ficam as processadas,
onde está o cache delas no Drive e para que servem (`papel`):

- "treino": as 53.576 fotos do "test" da competição de 2015 (rótulos
  publicados depois pelos organizadores) e os datasets externos do
  experimento (`DATASETS_EXTERNOS`: DDR, APTOS, DeepDRiD, IDRiD). Entram
  só no treino; validação e teste continuam os do EyePACS;
- "validacao_externa": bases que nunca entram no treino e só medem o
  modelo final (Messidor-2, mBRSET; ver `docs/DATASETS.md`).

`preparar_fonte` faz, para qualquer uma, o que antes estava repetido por
fonte: restaura o cache do Drive se existir; senão extrai as fotos
originais, processa com o mesmo pré-processamento do EyePACS, grava o
cache (só no notebook Preparar_Caches) e confere que as imagens
processadas estão no disco.

O train do EyePACS, de onde saem treino, validação e teste, não é uma
fonte daqui: é preparado por `execucao.preparar_imagens_eyepacs`.
"""

import os
import shutil
import zipfile
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from . import (
    ambiente_colab,
    configuracao,
    conjunto_dados,
    datasets_externos,
    pre_processamento,
)

PAPEIS = ("treino", "validacao_externa")


@dataclass(frozen=True)
class FonteDados:
    """
    Uma fonte de imagens: `ler_tabela(dados)` monta a tabela de rótulos
    no formato do treino (`dados` é a tabela do train do EyePACS, usada
    para conferir que nenhum paciente se repete); `extrair(tabela)` põe
    as fotos originais no disco e devolve a pasta delas;
    `coluna_entrada` é a coluna com o caminho de cada foto original
    quando ele não é o próprio `image`.
    """

    nome: str
    papel: str
    pasta_processada: str
    caminhos_zip_cache: tuple
    caminho_lista_suspeitas: str
    ler_tabela: Callable
    extrair: Callable
    coluna_entrada: Optional[str] = None
    fracao_maxima_falhas: float = 0.0
    apagar_originais: bool = False

    def __post_init__(self):
        if self.papel not in PAPEIS:
            raise ValueError(f"Papel desconhecido para {self.nome}: {self.papel}")


# ============================================================
# AS FONTES
# ============================================================


def _extrair_extra(tabela):
    """
    Fotos do "test" de 2015, dos pedaços do zip principal.
    """

    # Libera os pedaços do train, que já foram extraídos
    ambiente_colab.apagar_pedacos("train")

    ambiente_colab.extrair_imagens(
        "test",
        configuracao.PASTA_IMAGENS_EXTRA,
        nomes_esperados=tabela["image"].tolist()
    )

    return configuracao.PASTA_IMAGENS_EXTRA


def fonte_extra():
    """
    As 53.576 fotos do "test" da competição de 2015, só para o treino.
    """

    return FonteDados(
        nome="EyePACS_extra",
        papel="treino",
        pasta_processada=configuracao.PASTA_IMAGENS_OTIMIZADAS_EXTRA,
        caminhos_zip_cache=tuple(configuracao.CAMINHOS_ZIP_CACHE_EXTRA),
        caminho_lista_suspeitas=configuracao.CAMINHO_LISTA_SUSPEITAS_EXTRA,
        ler_tabela=conjunto_dados.carregar_dados_extra,
        extrair=_extrair_extra
    )


def _extrator_do_zip(nome):
    """
    Extração das fotos de um dataset externo: só as da tabela, de dentro
    do zip baixado para o Drive.
    """

    def extrair(tabela):
        pasta_originais = os.path.join(configuracao.PASTA_EXTERNOS_BRUTOS, nome)

        print(f"Extraindo {len(tabela)} fotos do {nome}...")

        with zipfile.ZipFile(configuracao.ZIPS_DATASETS_EXTERNOS[nome]) as arquivo_zip:
            arquivo_zip.extractall(
                pasta_originais,
                members=tabela["membro"].tolist()
            )

        return pasta_originais

    return extrair


def fonte_externa(nome):
    """
    Um dataset externo do treino (`configuracao.ZIPS_DATASETS_EXTERNOS`).
    Até `FRACAO_MAXIMA_FALHAS_EXTERNOS` das fotos podem falhar no
    pré-processamento: ficam fora do treino em vez de parar a execução.
    """

    return FonteDados(
        nome=nome,
        papel="treino",
        pasta_processada=configuracao.PASTAS_IMAGENS_EXTERNAS[nome],
        caminhos_zip_cache=tuple(configuracao.CAMINHOS_ZIP_CACHE_EXTERNOS[nome]),
        caminho_lista_suspeitas=configuracao.CAMINHOS_LISTA_SUSPEITAS_EXTERNOS[nome],
        ler_tabela=lambda dados: datasets_externos.carregar_dataset_externo(nome),
        extrair=_extrator_do_zip(nome),
        coluna_entrada="membro",
        fracao_maxima_falhas=configuracao.FRACAO_MAXIMA_FALHAS_EXTERNOS,
        apagar_originais=True
    )


def fontes_de_treino():
    """
    As fontes que o experimento ativo põe no treino, além do EyePACS:
    as extras de 2015 (`USAR_DADOS_EXTRA`) e os datasets externos
    (`DATASETS_EXTERNOS`), nessa ordem.
    """

    fontes = [fonte_extra()] if configuracao.USAR_DADOS_EXTRA else []

    fontes += [fonte_externa(nome) for nome in configuracao.DATASETS_EXTERNOS]

    return fontes


# ============================================================
# O PREPARO
# ============================================================


def preparar_fonte(fonte, dados=None, tabela=None):
    """
    Deixa no disco as imagens processadas de `fonte`: restaura o cache do
    Drive se existir; senão extrai as fotos originais, processa, grava o
    cache (só com RETINOPATIA_GRAVAR_CACHE=1) e, se a fonte pede, apaga
    as originais. `tabela` (padrão: `fonte.ler_tabela(dados)`) evita ler
    os rótulos de novo.

    Retorna a tabela só com as imagens processadas: as que falharam (até
    `fonte.fracao_maxima_falhas`) ficam de fora; acima disso, o cache
    está incompleto e a execução para.
    """

    if tabela is None:
        tabela = fonte.ler_tabela(dados)

    if not ambiente_colab.restaurar_cache_imagens(
        fonte.pasta_processada,
        list(fonte.caminhos_zip_cache)
    ):
        pasta_originais = fonte.extrair(tabela)

        pre_processamento.pre_processar_dataset(
            tabela,
            pasta_imagens=pasta_originais,
            pasta_saida=fonte.pasta_processada,
            caminho_lista_suspeitas=fonte.caminho_lista_suspeitas,
            coluna_entrada=fonte.coluna_entrada,
            fracao_maxima_falhas=fonte.fracao_maxima_falhas
        )

        ambiente_colab.salvar_cache_imagens(
            fonte.pasta_processada,
            fonte.caminhos_zip_cache[0]
        )

        if fonte.apagar_originais:
            shutil.rmtree(pasta_originais, ignore_errors=True)

    return datasets_externos.manter_processadas(
        tabela,
        fonte.pasta_processada,
        fonte.fracao_maxima_falhas
    )


def preparar_dados_adicionais(dados):
    """
    As fotos que entram só no treino (`fontes_de_treino`), já
    processadas no disco, numa tabela para `conjunto_dados.ampliar_treino`.
    Uma fonte de validação externa aqui é erro: ela nunca entra no treino.

    Retorna a tabela, ou None se o experimento não usa nenhuma.
    """

    partes = []

    for fonte in fontes_de_treino():
        if fonte.papel != "treino":
            raise ValueError(
                f"{fonte.nome} é de {fonte.papel} e não pode entrar no treino."
            )

        partes.append(
            preparar_fonte(fonte, dados).drop(columns="membro", errors="ignore")
        )

    if not partes:
        return None

    return pd.concat(partes, ignore_index=True)


def preparar_caches():
    """
    Gera os caches de imagens processadas do tamanho do experimento
    ativo (datasets externos, train e extras) e grava os zips na pasta
    compartilhada. É a função do notebook Preparar_Caches, rodado com o
    Drive pessoal do responsável montado e RETINOPATIA_GRAVAR_CACHE=1.

    Processa todas as imagens do train (não só as do treino do
    projeto), para que o mesmo cache sirva a qualquer experimento com
    esse tamanho. Um cache que já existe no Drive não é refeito. Os
    externos vêm primeiro: são pequenos e ficam salvos mesmo se a sessão
    cair no meio do EyePACS (~88 GB de fotos originais).
    """

    if not configuracao.GRAVAR_CACHE:
        raise RuntimeError(
            "Defina RETINOPATIA_GRAVAR_CACHE=1 (notebook Preparar_Caches) "
            "antes de importar o pacote."
        )

    print(
        f"Caches de {configuracao.TAMANHO_IMAGEM[0]} px "
        f"(sigma de Ben {configuracao.SIGMA_BEN:.2f}, erosão "
        f"{configuracao.EROSAO_MASCARA}) em "
        f"{configuracao.PASTA_CACHE_COMPARTILHADA}"
    )

    for nome in configuracao.DATASETS_EXTERNOS:
        fonte = fonte_externa(nome)

        if os.path.isfile(fonte.caminhos_zip_cache[0]):
            print(f"O cache do {nome} já existe: {fonte.caminhos_zip_cache[0]}")
            continue

        preparar_fonte(fonte)

        # Libera o disco para o EyePACS
        shutil.rmtree(fonte.pasta_processada, ignore_errors=True)

    ambiente_colab.extrair_rotulos()

    dados = conjunto_dados.carregar_csv()
    dados = conjunto_dados.criar_classe_binaria(dados)
    dados = conjunto_dados.extrair_identificador_paciente(dados)

    caminho_cache_train = configuracao.CAMINHOS_ZIP_CACHE[0]

    if os.path.isfile(caminho_cache_train):
        print(f"O cache do train já existe: {caminho_cache_train}")

    else:
        ambiente_colab.extrair_imagens(
            "train",
            configuracao.PASTA_IMAGENS,
            nomes_esperados=dados["image"].tolist()
        )

        pre_processamento.pre_processar_dataset(dados)

        ambiente_colab.salvar_cache_imagens()

        # As fotos originais do train não são mais necessárias e ocupam
        # ~35 GB que a extração do test vai precisar
        shutil.rmtree(configuracao.PASTA_IMAGENS, ignore_errors=True)

    if configuracao.USAR_DADOS_EXTRA:
        fonte = fonte_extra()

        if os.path.isfile(fonte.caminhos_zip_cache[0]):
            print(
                "O cache das imagens extras já existe: "
                f"{fonte.caminhos_zip_cache[0]}"
            )

        else:
            preparar_fonte(fonte, dados)

    print("Caches prontos.")
