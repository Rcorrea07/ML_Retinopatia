"""
Cola específica do Google Colab.

Este módulo concentra tudo que só funciona dentro do Colab: montagem do
Google Drive, extração do dataset (train e, para o treino ampliado, o
test de 2015 e os datasets externos), cache das imagens processadas no
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
import zipfile

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


def _extrair_imagens(
    nome,
    pasta_imagens,
    caminho_zip_principal=configuracao.CAMINHO_ZIP_PRINCIPAL,
    pasta_base=configuracao.PASTA_BASE,
    nomes_esperados=None
):
    """
    Extrai as imagens de uma parte do dataset ("train" ou "test").

    No zip principal cada parte é um zip dividido em pedaços de ~8 GB
    (nome.zip.001, .002, ...). O 7z lê os pedaços em sequência, sem
    juntá-los num arquivo único (o cat antigo duplicava ~35 GB no
    disco), e os pedaços são apagados depois da extração.
    """

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
    caminhos_zip_cache=configuracao.CAMINHOS_ZIP_CACHE_EXTRA,
    caminho_lista_suspeitas=configuracao.CAMINHO_LISTA_SUSPEITAS_EXTRA
):
    """
    Deixa prontas as imagens extras (test de 2015) já processadas: usa o
    cache do Drive se existir; senão extrai as fotos originais, processa
    com o mesmo pré-processamento do train e, no notebook Preparar_Caches,
    salva o cache no Drive.
    """

    from . import pre_processamento

    if restaurar_cache_imagens(pasta_saida, caminhos_zip_cache):
        return

    # Libera os pedaços do train, que já foram extraídos
    _executar(
        f'rm -f "{configuracao.PASTA_BASE}/train.zip."0* '
        f'"{configuracao.PASTA_BASE}/train_completo.zip"'
    )

    _extrair_imagens(
        "test",
        pasta_imagens,
        nomes_esperados=df_extra["image"].tolist()
    )

    pre_processamento.pre_processar_dataset(
        df_extra,
        pasta_imagens=pasta_imagens,
        pasta_saida=pasta_saida,
        caminho_lista_suspeitas=caminho_lista_suspeitas
    )

    salvar_cache_imagens(pasta_saida, caminhos_zip_cache[0])


def preparar_imagens_externas(nome, tabela):
    """
    Deixa prontas as fotos processadas de um dataset externo
    (`datasets_externos.carregar_dataset_externo`): usa o cache do Drive
    se existir; senão tira do zip só as fotos da tabela, processa com o
    mesmo pré-processamento do EyePACS e, no notebook Preparar_Caches,
    salva o cache no Drive.

    Retorna a tabela só com as fotos processadas: as que falharam (até
    `FRACAO_MAXIMA_FALHAS_EXTERNOS`) ficam fora do treino.
    """

    from . import datasets_externos, pre_processamento

    pasta_saida = configuracao.PASTAS_IMAGENS_EXTERNAS[nome]
    caminhos_zip_cache = configuracao.CAMINHOS_ZIP_CACHE_EXTERNOS[nome]

    if not restaurar_cache_imagens(pasta_saida, caminhos_zip_cache):
        pasta_bruta = os.path.join(configuracao.PASTA_EXTERNOS_BRUTOS, nome)

        print(f"Extraindo {len(tabela)} fotos do {nome}...")

        caminho_zip = configuracao.ZIPS_DATASETS_EXTERNOS[nome]

        with zipfile.ZipFile(caminho_zip) as arquivo_zip:
            arquivo_zip.extractall(
                pasta_bruta,
                members=tabela["membro"].tolist()
            )

        pre_processamento.pre_processar_dataset(
            tabela,
            pasta_imagens=pasta_bruta,
            pasta_saida=pasta_saida,
            caminho_lista_suspeitas=(
                configuracao.CAMINHOS_LISTA_SUSPEITAS_EXTERNOS[nome]
            ),
            coluna_entrada="membro",
            fracao_maxima_falhas=configuracao.FRACAO_MAXIMA_FALHAS_EXTERNOS
        )

        salvar_cache_imagens(pasta_saida, caminhos_zip_cache[0])

        shutil.rmtree(pasta_bruta, ignore_errors=True)

    return datasets_externos.manter_processadas(tabela, pasta_saida)


def preparar_dados_adicionais(dados):
    """
    Fotos que entram só no treino, já processadas no disco: as extras de
    2015 (`USAR_DADOS_EXTRA`) e os datasets externos do experimento
    (`DATASETS_EXTERNOS`). Cada parte usa o seu cache do Drive se existir,
    senão é processada nesta sessão.

    Retorna a tabela delas, para `conjunto_dados.ampliar_treino`, ou None
    se o experimento não usa nenhuma.
    """

    import pandas as pd

    from . import conjunto_dados, datasets_externos

    partes = []

    if configuracao.USAR_DADOS_EXTRA:
        df_extra = conjunto_dados.carregar_dados_extra(dados)

        preparar_imagens_extra(df_extra)

        partes.append(df_extra)

    for nome in configuracao.DATASETS_EXTERNOS:
        tabela = datasets_externos.carregar_dataset_externo(nome)

        partes.append(
            preparar_imagens_externas(nome, tabela).drop(columns="membro")
        )

    if not partes:
        return None

    return pd.concat(partes, ignore_index=True)


def restaurar_cache_imagens(
    pasta_imagens_otimizadas=configuracao.PASTA_IMAGENS_OTIMIZADAS,
    caminhos_zip_cache=configuracao.CAMINHOS_ZIP_CACHE
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
    pasta_imagens_otimizadas=configuracao.PASTA_IMAGENS_OTIMIZADAS,
    caminho_zip_cache=configuracao.CAMINHO_ZIP_CACHE,
    gravar=configuracao.GRAVAR_CACHE
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

    from . import conjunto_dados, datasets_externos, pre_processamento

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
        caminho_zip_cache = configuracao.CAMINHOS_ZIP_CACHE_EXTERNOS[nome][0]

        if os.path.isfile(caminho_zip_cache):
            print(f"O cache do {nome} já existe: {caminho_zip_cache}")
            continue

        preparar_imagens_externas(
            nome,
            datasets_externos.carregar_dataset_externo(nome)
        )

        # Libera o disco para o EyePACS
        shutil.rmtree(
            configuracao.PASTAS_IMAGENS_EXTERNAS[nome],
            ignore_errors=True
        )

    extrair_rotulos()

    dados = conjunto_dados.carregar_csv()
    dados = conjunto_dados.criar_classe_binaria(dados)
    dados = conjunto_dados.extrair_identificador_paciente(dados)

    if os.path.isfile(configuracao.CAMINHO_ZIP_CACHE):
        print(f"O cache do train já existe: {configuracao.CAMINHO_ZIP_CACHE}")

    else:
        _extrair_imagens(
            "train",
            configuracao.PASTA_IMAGENS,
            nomes_esperados=dados["image"].tolist()
        )

        pre_processamento.pre_processar_dataset(dados)

        salvar_cache_imagens()

        # As fotos originais do train não são mais necessárias e ocupam
        # ~35 GB que a extração do test vai precisar
        shutil.rmtree(configuracao.PASTA_IMAGENS, ignore_errors=True)

    if configuracao.USAR_DADOS_EXTRA:
        df_extra = conjunto_dados.carregar_dados_extra(dados)

        if os.path.isfile(configuracao.CAMINHO_ZIP_CACHE_EXTRA):
            print(
                "O cache das imagens extras já existe: "
                f"{configuracao.CAMINHO_ZIP_CACHE_EXTRA}"
            )

        else:
            preparar_imagens_extra(df_extra)

    print("Caches prontos.")


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


def baixar_datasets_externos(pasta=configuracao.PASTA_DATASETS):
    """
    Baixa os datasets de `configuracao.DATASETS_KAGGLE` e
    `configuracao.DATASETS_URL`, cada um na sua subpasta de `pasta`.
    Pula os que já estão lá, então pode rodar de novo se cair no meio.
    """

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
