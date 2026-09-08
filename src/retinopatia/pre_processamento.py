"""
Pré-processamento das imagens.

O recorte da retina usa o método de Otsu para encontrar a região útil da
foto, descartando a moldura preta. Esse recorte é feito uma única vez,
de forma offline, e o resultado é guardado em disco — no notebook
original isso evitava pagar o custo do recorte a cada época (que chegava
a 40 minutos por época).

A mesma função de recorte (`recortar_otsu`) é usada tanto pela inspeção
visual quanto pelo processamento em lote.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from . import configuracao


def criar_mascara_otsu(img):
    """
    Cria a máscara binária da retina usando o método de Otsu.
    """

    # Converte para escala de cinza somente para criar a máscara
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    # Suaviza pequenos ruídos da máscara.
    # A imagem original não é desfocada.
    gray_blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Cria a máscara usando o método de Otsu
    _, thresh = cv2.threshold(
        gray_blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return thresh


def recortar_otsu(img):
    """
    Recorta a imagem na caixa delimitadora da máscara de Otsu.

    Retorna:
        (recorte, mascara, status)

    Onde status pode ser:
        "ok"               -> recorte válido
        "mascara_vazia"    -> Otsu não encontrou nenhum pixel
        "recorte_invalido" -> a caixa delimitadora não tem área útil

    Quando o status é diferente de "ok", o recorte devolvido é None.
    """

    thresh = criar_mascara_otsu(img)

    # Localiza os pixels pertencentes à região detectada
    pontos = cv2.findNonZero(thresh)

    # Se nenhum ponto for encontrado, registra a falha
    if pontos is None:
        return None, thresh, "mascara_vazia"

    # Calcula a caixa delimitadora da retina
    x, y, w, h = cv2.boundingRect(pontos)

    if w <= 0 or h <= 0:
        return None, thresh, "recorte_invalido"

    # Recorta a imagem original
    recortada = img[
        y:y + h,
        x:x + w
    ]

    if recortada.size == 0:
        return None, thresh, "recorte_invalido"

    return recortada, thresh, "ok"


def pre_visualizar_recorte(
    dados,
    pasta_imagens=configuracao.PASTA_IMAGENS,
    numero_amostras=configuracao.NUMERO_AMOSTRAS_INSPECAO,
    seed=configuracao.SEED
):
    """
    Inspeção visual: mostra, para uma amostra de imagens, a foto
    original, a máscara de Otsu e o recorte resultante.
    """

    amostra = dados.sample(
        n=min(numero_amostras, len(dados)),
        random_state=seed
    )

    for nome in amostra["image"]:
        caminho = os.path.join(pasta_imagens, nome)
        img = cv2.imread(caminho)

        if img is None:
            print(f"Erro ao ler: {nome}")
            continue

        recorte, thresh, status = recortar_otsu(img)

        if status == "mascara_vazia":
            print(f"Máscara vazia para a imagem: {nome}")
            recorte = img

        elif status != "ok":
            print(f"Recorte inválido para a imagem: {nome}")
            recorte = img

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(thresh, cmap="gray")
        plt.title("Máscara de Otsu")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB))
        plt.title("Recorte")
        plt.axis("off")

        plt.tight_layout()
        plt.show()


def processar_e_salvar_imagem(
    img_nome,
    pasta_imagens,
    pasta_saida,
    tamanho_imagem
):
    """
    Lê, recorta, redimensiona e salva uma imagem.

    Retorna:
        (nome_da_imagem, status_do_processamento)
    """

    caminho_in = os.path.join(
        pasta_imagens,
        img_nome
    )

    caminho_out = os.path.join(
        pasta_saida,
        img_nome
    )

    # Se a imagem já foi processada, não processa novamente
    if os.path.exists(caminho_out):
        return img_nome, "existente"

    # Tenta ler a imagem
    img = cv2.imread(caminho_in)

    if img is None:
        return img_nome, "erro_leitura"

    recortada, _, status = recortar_otsu(img)

    if status != "ok":
        return img_nome, status

    # Redimensiona para o tamanho esperado pela EfficientNetB0
    img_final = cv2.resize(
        recortada,
        tamanho_imagem,
        interpolation=cv2.INTER_AREA
    )

    # Salva a imagem e guarda o resultado da operação
    sucesso = cv2.imwrite(
        caminho_out,
        img_final
    )

    if not sucesso:
        return img_nome, "erro_escrita"

    return img_nome, "processada"


def pre_processar_dataset(
    dados,
    pasta_imagens=configuracao.PASTA_IMAGENS,
    pasta_saida=configuracao.PASTA_IMAGENS_OTIMIZADAS,
    tamanho_imagem=configuracao.TAMANHO_IMAGEM,
    maximo_workers=configuracao.MAXIMO_WORKERS
):
    """
    Recorta e redimensiona todas as imagens do dataset em paralelo,
    salvando o resultado em `pasta_saida`.

    A execução é interrompida com RuntimeError se alguma imagem falhar
    ou se alguma imagem esperada não estiver na pasta final.

    Retorna o DataFrame com o status de cada imagem.
    """

    os.makedirs(
        pasta_saida,
        exist_ok=True
    )

    print("Iniciando recorte otimizado em paralelo...")

    # Limita o número de threads para evitar sobrecarga
    numero_workers = min(
        maximo_workers,
        os.cpu_count() or 1
    )

    processar = partial(
        processar_e_salvar_imagem,
        pasta_imagens=pasta_imagens,
        pasta_saida=pasta_saida,
        tamanho_imagem=tamanho_imagem
    )

    with ThreadPoolExecutor(
        max_workers=numero_workers
    ) as executor:

        resultados = list(
            tqdm(
                executor.map(
                    processar,
                    dados["image"].values
                ),
                total=len(dados)
            )
        )

    print("\nProcessamento concluído!")

    # Organiza os resultados em uma tabela
    resultados_df = pd.DataFrame(
        resultados,
        columns=["image", "status"]
    )

    print("\nResumo do processamento:")
    print(
        resultados_df["status"]
        .value_counts()
    )

    falhas = resultados_df[
        resultados_df["status"].isin(configuracao.STATUS_DE_ERRO)
    ]

    # Interrompe a execução caso alguma imagem tenha falhado
    if not falhas.empty:
        print("\nExemplos de imagens que apresentaram problemas:")
        print(
            falhas.head(20).to_string(index=False)
        )

        raise RuntimeError(
            f"{len(falhas)} imagens falharam no pré-processamento."
        )

    # Confirma se todas as imagens esperadas existem na pasta final
    imagens_ausentes = [
        nome
        for nome in dados["image"].values
        if not os.path.isfile(
            os.path.join(
                pasta_saida,
                nome
            )
        )
    ]

    if imagens_ausentes:
        print("\nExemplos de imagens ausentes:")
        print(imagens_ausentes[:20])

        raise RuntimeError(
            f"{len(imagens_ausentes)} imagens não foram encontradas "
            "na pasta processada."
        )

    print(
        "\nTodas as imagens foram processadas "
        "ou já existiam corretamente!"
    )

    return resultados_df
