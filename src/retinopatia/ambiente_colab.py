"""
Cola específica do Google Colab e operações de arquivo.

Este módulo concentra tudo que só funciona dentro do Colab: montagem do
Google Drive, extração do zip principal do EyePACS (rótulos e as partes
"train" e "test"), cache das imagens processadas no Drive, downloads e
o desligamento da sessão. Ele não conhece tabelas de dados: o preparo
de cada fonte de imagens (extras de 2015, datasets externos, validação
externa) está em `fontes`. O restante do pacote não depende do Colab.

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


def desligar_ambiente():
    """
    Encerra a sessão do Colab e libera a GPU. Uma sessão parada depois do
    fim do treino continua gastando unidades de computação.
    """

    from google.colab import runtime

    print("Desligando o ambiente de execução...")
    runtime.unassign()


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
    caminho_zip_principal=None,
    pasta_base=None,
    caminho_csv=None
):
    """
    Extrai do zip principal apenas o CSV de rótulos. Basta quando as
    imagens processadas vêm do cache no Drive.
    """

    if caminho_zip_principal is None:
        caminho_zip_principal = configuracao.CAMINHO_ZIP_PRINCIPAL

    if pasta_base is None:
        pasta_base = configuracao.PASTA_BASE

    if caminho_csv is None:
        caminho_csv = configuracao.CAMINHO_CSV

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


def _imagens_completas(pasta_imagens, nomes_esperados=None):
    """
    Com `nomes_esperados`, True só se todas essas imagens existirem e
    não estiverem vazias; sem a lista, True se a pasta tiver algum
    arquivo (comportamento antigo).
    """

    if nomes_esperados is None:
        return os.path.isdir(pasta_imagens) and bool(os.listdir(pasta_imagens))

    return all(
        os.path.isfile(os.path.join(pasta_imagens, nome))
        and os.path.getsize(os.path.join(pasta_imagens, nome)) > 0
        for nome in nomes_esperados
    )


def apagar_pedacos(nome, pasta_base=None):
    """
    Apaga do disco local os pedaços de uma parte do zip principal
    (`nome.zip.001`, ...), depois que as imagens dela foram extraídas.
    """

    if pasta_base is None:
        pasta_base = configuracao.PASTA_BASE

    _executar(
        f'rm -f "{pasta_base}/{nome}.zip."0* '
        f'"{pasta_base}/{nome}_completo.zip"'
    )


def extrair_imagens(
    nome,
    pasta_imagens,
    caminho_zip_principal=None,
    pasta_base=None,
    nomes_esperados=None
):
    """
    Extrai as imagens de uma parte do dataset ("train" ou "test").

    No zip principal cada parte é um zip dividido em pedaços de ~8 GB
    (nome.zip.001, .002, ...). O 7z lê os pedaços em sequência, sem
    juntá-los num arquivo único (o cat antigo duplicava ~35 GB no
    disco), e os pedaços são apagados depois da extração.
    """

    if caminho_zip_principal is None:
        caminho_zip_principal = configuracao.CAMINHO_ZIP_PRINCIPAL

    if pasta_base is None:
        pasta_base = configuracao.PASTA_BASE

    if _imagens_completas(pasta_imagens, nomes_esperados):
        print(f"As imagens de '{nome}' já foram extraídas.")
        return

    os.makedirs(
        pasta_base,
        exist_ok=True
    )

    primeira_parte = os.path.join(pasta_base, f"{nome}.zip.001")

    if nomes_esperados is not None or not os.path.isfile(primeira_parte):
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

    # Com a lista de nomes, uma extração interrompida não passa por
    # completa só porque a pasta tem algum arquivo
    if nomes_esperados is not None and not _imagens_completas(
        pasta_imagens,
        nomes_esperados
    ):
        raise RuntimeError(
            f"Extração de '{nome}' incompleta; os pedaços foram mantidos "
            "para diagnóstico."
        )

    apagar_pedacos(nome, pasta_base)

    print(f"Imagens de '{nome}' extraídas em: {pasta_imagens}")


def extrair_7z(primeira_parte, destino):
    """
    Extrai um zip em várias partes (`.001`, `.002`, ...) lendo-as em
    sequência, direto do Drive, para `destino`. Pula se o destino já tem
    arquivos.
    """

    if os.path.isdir(destino) and os.listdir(destino):
        print(f"Já extraído em: {destino}")
        return

    os.makedirs(destino, exist_ok=True)

    print(f"Extraindo {primeira_parte}...")
    _executar(f'7z x -y -bd -o"{destino}" "{primeira_parte}" > /dev/null')


def extrair_dataset(
    caminho_zip_principal=None,
    pasta_base=None,
    pasta_imagens=None,
    somente_rotulos=False
):
    """
    Extrai o CSV de rótulos e as imagens do "train" para o disco local
    do Colab. Cada etapa é pulada se o resultado já existir. Com
    `somente_rotulos=True` (imagens processadas restauradas do cache),
    extrai só o CSV.
    """

    if caminho_zip_principal is None:
        caminho_zip_principal = configuracao.CAMINHO_ZIP_PRINCIPAL

    if pasta_base is None:
        pasta_base = configuracao.PASTA_BASE

    if pasta_imagens is None:
        pasta_imagens = configuracao.PASTA_IMAGENS

    extrair_rotulos(caminho_zip_principal, pasta_base)

    if somente_rotulos:
        return

    extrair_imagens(
        "train",
        pasta_imagens,
        caminho_zip_principal,
        pasta_base
    )


def restaurar_cache_imagens(
    pasta_imagens_otimizadas=None,
    caminhos_zip_cache=None
):
    """
    Restaura do Drive as imagens já processadas desta versão do
    pré-processamento e deste tamanho. Procura o zip em cada caminho de
    `caminhos_zip_cache`, na ordem (primeiro a pasta compartilhada da
    conta pessoal, depois a pasta de cache do projeto).

    Retorna True se as imagens processadas estão disponíveis (já
    estavam no disco ou foram restauradas), e False se será preciso
    extrair o dataset bruto e processá-lo.
    """

    if pasta_imagens_otimizadas is None:
        pasta_imagens_otimizadas = configuracao.PASTA_IMAGENS_OTIMIZADAS

    if caminhos_zip_cache is None:
        caminhos_zip_cache = configuracao.CAMINHOS_ZIP_CACHE

    if (
        os.path.isdir(pasta_imagens_otimizadas)
        and os.listdir(pasta_imagens_otimizadas)
    ):
        print("As imagens processadas já estão no disco do Colab.")
        return True

    if isinstance(caminhos_zip_cache, str):
        caminhos_zip_cache = [caminhos_zip_cache]

    caminho_zip_cache = next(
        (caminho for caminho in caminhos_zip_cache if os.path.isfile(caminho)),
        None
    )

    if caminho_zip_cache is None:
        print(
            "Nenhum cache de imagens processadas no Drive. Procurado em:\n  "
            + "\n  ".join(caminhos_zip_cache)
        )

        if not configuracao.GRAVAR_CACHE:
            print(
                "As imagens serão processadas só neste Colab (cerca de 1 h "
                "com os dados extras). Para os próximos treinos, gere o "
                "cache uma vez com o notebook Preparar_Caches."
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
    pasta_imagens_otimizadas=None,
    caminho_zip_cache=None,
    gravar=None
):
    """
    Guarda no Drive um zip das imagens processadas, se ainda não
    existir. O zip é montado no disco local e só depois copiado, porque
    escrever arquivo a arquivo no Drive é lento. Os JPEGs não são
    recomprimidos (zip -0).

    Só grava com `gravar=True` (RETINOPATIA_GRAVAR_CACHE=1, usado pelo
    notebook Preparar_Caches com o Drive pessoal montado). Nos treinos, a
    conta do projeto não tem espaço para caches novos, e gravar na pasta
    compartilhada gastaria a cota dela.
    """

    if pasta_imagens_otimizadas is None:
        pasta_imagens_otimizadas = configuracao.PASTA_IMAGENS_OTIMIZADAS

    if caminho_zip_cache is None:
        caminho_zip_cache = configuracao.CAMINHOS_ZIP_CACHE[0]

    if gravar is None:
        gravar = configuracao.GRAVAR_CACHE

    if not gravar:
        print(
            "Cache não gravado no Drive (só o notebook Preparar_Caches "
            "grava caches)."
        )
        return

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


def _credenciais_kaggle():
    """
    Passa ao ambiente as credenciais do Kaggle guardadas nos segredos do
    Colab (ícone de chave na barra lateral): KAGGLE_API_TOKEN, o token
    novo de Kaggle → Settings → API, ou o par KAGGLE_USERNAME e
    KAGGLE_KEY do kaggle.json antigo.
    """

    from google.colab import userdata

    encontradas = []

    for nome in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        try:
            os.environ[nome] = userdata.get(nome)
            encontradas.append(nome)
        except (userdata.SecretNotFoundError, userdata.NotebookAccessError):
            pass

    if not encontradas:
        raise RuntimeError(
            "Nenhuma credencial do Kaggle nos segredos do Colab. Crie o "
            "segredo KAGGLE_API_TOKEN (chave na barra lateral) com o token "
            "de Kaggle → Settings → API e ative o acesso deste notebook."
        )


def baixar_do_kaggle(identificador, pasta_destino):
    """
    Baixa um dataset do Kaggle direto para uma pasta do Drive, sem
    descompactar (o zip fica como veio).
    """

    _credenciais_kaggle()

    # A versão do Colab pode não conhecer o token novo
    _executar("pip install -q -U kaggle")

    os.makedirs(pasta_destino, exist_ok=True)

    _executar(f'kaggle datasets download -d {identificador} -p "{pasta_destino}"')


def baixar_url_para_drive(url, caminho_destino):
    """
    Baixa um arquivo de um endereço público direto para o Drive.
    """

    os.makedirs(os.path.dirname(caminho_destino), exist_ok=True)

    _executar(f'curl -L --fail -sS -o "{caminho_destino}" "{url}"')


def baixar_datasets_externos(pasta=None):
    """
    Baixa os datasets de `configuracao.DATASETS_KAGGLE` e
    `configuracao.DATASETS_URL`, cada um na sua subpasta de `pasta`.
    Pula os que já estão lá, então pode rodar de novo se cair no meio.
    """

    if pasta is None:
        pasta = configuracao.PASTA_DATASETS

    for nome, identificador in configuracao.DATASETS_KAGGLE.items():
        destino = os.path.join(pasta, nome)

        if os.path.isdir(destino) and os.listdir(destino):
            print(f"{nome}: já baixado em {destino}")
            continue

        print(f"{nome}: baixando {identificador} do Kaggle...")
        baixar_do_kaggle(identificador, destino)

    for nome, (arquivo, url) in configuracao.DATASETS_URL.items():
        destino = os.path.join(pasta, nome, arquivo)

        if os.path.isfile(destino) and os.path.getsize(destino) > 0:
            print(f"{nome}: já baixado em {destino}")
            continue

        print(f"{nome}: baixando {url}...")
        baixar_url_para_drive(url, destino)

    print("\nNa pasta de datasets:")

    for nome in sorted(os.listdir(pasta)):
        caminho = os.path.join(pasta, nome)
        arquivos = (
            [os.path.join(caminho, f) for f in os.listdir(caminho)]
            if os.path.isdir(caminho) else [caminho]
        )
        tamanho = sum(os.path.getsize(f) for f in arquivos if os.path.isfile(f))
        print(f"  {nome}: {tamanho / 1e9:.2f} GB")


def baixar_arquivo(caminho):
    """
    Baixa um arquivo do Colab para a máquina do usuário.
    """

    from google.colab import files

    files.download(caminho)
