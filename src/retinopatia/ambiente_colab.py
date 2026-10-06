"""
Cola específica do Google Colab.

Este módulo concentra tudo que só funciona dentro do Colab: montagem do
Google Drive, extração do dataset, cache das imagens processadas no
Drive e download de arquivos. O restante do pacote não depende do Colab.

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


def extrair_dataset(
    caminho_zip_principal=configuracao.CAMINHO_ZIP_PRINCIPAL,
    pasta_base=configuracao.PASTA_BASE,
    pasta_imagens=configuracao.PASTA_IMAGENS,
    somente_rotulos=False
):
    """
    Extrai o dataset para o disco local do Colab.

    A extração é inteligente: se a pasta de imagens já existir, a etapa
    inteira é pulada para economizar tempo. Com `somente_rotulos=True`
    (imagens processadas restauradas do cache), extrai só o CSV.
    """

    if somente_rotulos:
        extrair_rotulos(caminho_zip_principal, pasta_base)
        return

    print(" Verificando o ambiente de dados...")

    if not os.path.exists(pasta_imagens):
        os.makedirs(
            pasta_base,
            exist_ok=True
        )

        print("Extraindo ZIP principal...")
        _executar(
            f'unzip -q -o "{caminho_zip_principal}" -d "{pasta_base}"'
        )

        print("Extraindo CSV de labels...")
        _executar(
            f'unzip -q -o "{pasta_base}/trainLabels.csv.zip" -d "{pasta_base}"'
        )

        print(
            "Juntando e extraindo as imagens de treino "
            "(Isso vai levar alguns minutos)..."
        )
        _executar(
            f"cat {pasta_base}/train.zip.* > {pasta_base}/train_completo.zip"
        )
        _executar(
            f'unzip -q -o {pasta_base}/train_completo.zip -d "{pasta_base}"'
        )

        print(" Extração concluída!")

    else:
        print(
            " Os dados já foram extraídos anteriormente. "
            "Pulando esta etapa para economizar tempo."
        )


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
