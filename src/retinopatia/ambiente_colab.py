"""
Cola específica do Google Colab.

Este módulo concentra tudo que só funciona dentro do Colab: montagem do
Google Drive, extração do dataset e download de arquivos. O restante do
pacote não depende do Colab.

Observação: no notebook original a extração usava os atalhos de shell do
IPython (`!unzip`, `!cat`), que não existem em um arquivo `.py`. Aqui os
mesmos comandos são executados via `subprocess`, com `check=True` para
que uma extração malsucedida interrompa a execução em vez de falhar
silenciosamente mais adiante.
"""

import os
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


def extrair_dataset(
    caminho_zip_principal=configuracao.CAMINHO_ZIP_PRINCIPAL,
    pasta_base=configuracao.PASTA_BASE,
    pasta_imagens=configuracao.PASTA_IMAGENS
):
    """
    Extrai o dataset para o disco local do Colab.

    A extração é inteligente: se a pasta de imagens já existir, a etapa
    inteira é pulada para economizar tempo.
    """

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


def baixar_arquivo(caminho):
    """
    Baixa um arquivo do Colab para a máquina do usuário.
    """

    from google.colab import files

    files.download(caminho)
