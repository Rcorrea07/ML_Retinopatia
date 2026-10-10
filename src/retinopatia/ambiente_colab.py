"""
Cola específica do Google Colab.

Este módulo concentra tudo que só funciona dentro do Colab: montagem do
Google Drive, extração do dataset (train e, para o treino ampliado, o
test de 2015), cache das imagens processadas no Drive e download de
arquivos. O restante do pacote não depende do Colab.

Observação: no notebook original a extração usava os atalhos de shell do
IPython (`!unzip`, `!cat`), que não existem em um arquivo `.py`. Aqui os
mesmos comandos são executados via `subprocess`, com `check=True` para
que uma extração malsucedida interrompa a execução em vez de falhar
silenciosamente mais adiante.
"""

import os
import shutil
import subprocess

from . import configuracao


def montar_drive(ponto_de_montagem="/content/drive"):
    """
    Conecta o Google Drive ao ambiente do Colab.
    """

    from google.colab import drive

    drive.mount(ponto_de_montagem)


def _executar(comando):
    """
    Executa um comando de shell e interrompe em caso de erro.
    """

    subprocess.run(
        comando,
        shell=True,
        check=True
    )


def extrair_rotulos(
    caminho_zip_principal=configuracao.CAMINHO_ZIP_PRINCIPAL,
    pasta_base=configuracao.PASTA_BASE,
    caminho_csv=configuracao.CAMINHO_CSV
):
    """
    Extrai do zip principal apenas o CSV de rótulos. Basta quando as
    imagens processadas vêm do cache no Drive.
    """

    if os.path.isfile(caminho_csv):
        print("O CSV de rótulos já foi extraído.")
        return

    os.makedirs(
        pasta_base,
        exist_ok=True
    )

    print("Extraindo só o CSV de labels do ZIP principal...")
    _executar(
        f'unzip -q -o "{caminho_zip_principal}" trainLabels.csv.zip '
        f'-d "{pasta_base}"'
    )
    _executar(
        f'unzip -q -o "{pasta_base}/trainLabels.csv.zip" -d "{pasta_base}"'
    )


def _extrair_imagens(
    nome,
    pasta_imagens,
    caminho_zip_principal=configuracao.CAMINHO_ZIP_PRINCIPAL,
    pasta_base=configuracao.PASTA_BASE
):
    """
    Extrai as imagens de uma parte do dataset ("train" ou "test").

    No zip principal cada parte é um zip dividido em pedaços de ~8 GB
    (nome.zip.001, .002, ...). O 7z lê os pedaços em sequência, sem
    juntá-los num arquivo único (o cat antigo duplicava ~35 GB no
    disco), e os pedaços são apagados depois da extração.
    """

    if os.path.isdir(pasta_imagens) and os.listdir(pasta_imagens):
        print(f"As imagens de '{nome}' já foram extraídas.")
        return

    os.makedirs(
        pasta_base,
        exist_ok=True
    )

    primeira_parte = os.path.join(pasta_base, f"{nome}.zip.001")

    if not os.path.isfile(primeira_parte):
        print(f"Copiando os pedaços de '{nome}' do ZIP principal...")
        _executar(
            f'unzip -q -o "{caminho_zip_principal}" "{nome}.zip.*" '
            f'-d "{pasta_base}"'
        )

    print(
        f"Extraindo as imagens de '{nome}' "
        "(isso vai levar alguns minutos)..."
    )
    _executar(
        f'7z x -y -bd -o"{pasta_base}" "{primeira_parte}" > /dev/null'
    )

    _executar(
        f'rm -f "{pasta_base}/{nome}.zip."0* '
        f'"{pasta_base}/{nome}_completo.zip"'
    )

    print(f"Imagens de '{nome}' extraídas em: {pasta_imagens}")


def extrair_dataset(
    caminho_zip_principal=configuracao.CAMINHO_ZIP_PRINCIPAL,
    pasta_base=configuracao.PASTA_BASE,
    pasta_imagens=configuracao.PASTA_IMAGENS,
    somente_rotulos=False
):
    """
    Extrai o CSV de rótulos e as imagens do "train" para o disco local
    do Colab. Cada etapa é pulada se o resultado já existir. Com
    `somente_rotulos=True` (imagens processadas restauradas do cache),
    extrai só o CSV.
    """

    extrair_rotulos(caminho_zip_principal, pasta_base)

    if somente_rotulos:
        return

    _extrair_imagens(
        "train",
        pasta_imagens,
        caminho_zip_principal,
        pasta_base
    )


def preparar_imagens_extra(
    df_extra,
    pasta_imagens=configuracao.PASTA_IMAGENS_EXTRA,
    pasta_saida=configuracao.PASTA_IMAGENS_OTIMIZADAS_EXTRA,
    caminho_zip_cache=configuracao.CAMINHO_ZIP_CACHE_EXTRA,
    caminho_lista_suspeitas=configuracao.CAMINHO_LISTA_SUSPEITAS_EXTRA
):
    """
    Deixa prontas as imagens extras (test de 2015) já processadas: usa o
    cache do Drive se existir; senão extrai as fotos originais, processa
    com o mesmo pré-processamento do train e salva o cache no Drive.
    """

    from . import pre_processamento

    if restaurar_cache_imagens(pasta_saida, caminho_zip_cache):
        return

    # Libera os pedaços do train, que já foram extraídos
    _executar(
        f'rm -f "{configuracao.PASTA_BASE}/train.zip."0* '
        f'"{configuracao.PASTA_BASE}/train_completo.zip"'
    )

    _extrair_imagens("test", pasta_imagens)

    pre_processamento.pre_processar_dataset(
        df_extra,
        pasta_imagens=pasta_imagens,
        pasta_saida=pasta_saida,
        caminho_lista_suspeitas=caminho_lista_suspeitas
    )

    salvar_cache_imagens(pasta_saida, caminho_zip_cache)


def restaurar_cache_imagens(
    pasta_imagens_otimizadas=configuracao.PASTA_IMAGENS_OTIMIZADAS,
    caminho_zip_cache=configuracao.CAMINHO_ZIP_CACHE
):
    """
    Restaura do Drive as imagens já processadas desta versão do
    pré-processamento, se o zip existir.

    Retorna True se as imagens processadas estão disponíveis (já
    estavam no disco ou foram restauradas), e False se será preciso
    extrair o dataset bruto e processá-lo.
    """

    if (
        os.path.isdir(pasta_imagens_otimizadas)
        and os.listdir(pasta_imagens_otimizadas)
    ):
        print("As imagens processadas já estão no disco do Colab.")
        return True

    if not os.path.isfile(caminho_zip_cache):
        print(
            "Nenhum cache de imagens processadas no Drive "
            f"({caminho_zip_cache}). O dataset bruto será extraído."
        )
        return False

    print(f"Restaurando as imagens processadas de {caminho_zip_cache}...")

    _executar(
        f'unzip -q -o "{caminho_zip_cache}" '
        f'-d "{os.path.dirname(pasta_imagens_otimizadas)}"'
    )

    print("Cache restaurado!")

    return True


def salvar_cache_imagens(
    pasta_imagens_otimizadas=configuracao.PASTA_IMAGENS_OTIMIZADAS,
    caminho_zip_cache=configuracao.CAMINHO_ZIP_CACHE
):
    """
    Guarda no Drive um zip das imagens processadas, se ainda não
    existir. O zip é montado no disco local e só depois copiado, porque
    escrever arquivo a arquivo no Drive é lento. Os JPEGs não são
    recomprimidos (zip -0).
    """

    if os.path.isfile(caminho_zip_cache):
        print(f"O cache já existe no Drive: {caminho_zip_cache}")
        return

    pasta_pai, nome_pasta = os.path.split(
        pasta_imagens_otimizadas.rstrip("/")
    )

    zip_local = os.path.join("/content", os.path.basename(caminho_zip_cache))

    print("Compactando as imagens processadas...")
    _executar(
        f'cd "{pasta_pai}" && zip -q -r -0 "{zip_local}" "{nome_pasta}"'
    )

    os.makedirs(
        os.path.dirname(caminho_zip_cache),
        exist_ok=True
    )

    # Copia com outro nome e renomeia no fim: uma cópia interrompida
    # não deixa no Drive um zip incompleto com o nome definitivo
    print("Copiando o cache para o Drive...")
    shutil.copy(zip_local, caminho_zip_cache + ".parcial")
    os.replace(caminho_zip_cache + ".parcial", caminho_zip_cache)

    os.remove(zip_local)

    print(f"Cache salvo em: {caminho_zip_cache}")


def baixar_arquivo(caminho):
    """
    Baixa um arquivo do Colab para a máquina do usuário.
    """

    from google.colab import files

    files.download(caminho)
