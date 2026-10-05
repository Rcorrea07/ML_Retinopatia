"""
Pré-processamento das imagens (versão 2).

Segue a receita das soluções vencedoras das competições do Kaggle de
retinopatia (Ben Graham em 2015 e os notebooks de topo da APTOS 2019):

1. A máscara de Otsu localiza a retina e descarta a moldura preta.
2. Um quadrado centrado na retina, com lado igual ao diâmetro do
   círculo, é recortado e redimensionado. Assim toda retina ocupa a
   mesma área da imagem, independentemente da câmera.
3. A normalização de cor de Ben Graham remove diferenças de iluminação.
4. Uma máscara circular descarta a borda do círculo.

Esse processamento é feito uma única vez, de forma offline, e o
resultado é guardado em disco — no notebook original isso evitava pagar
o custo do recorte a cada época (que chegava a 40 minutos por época).

A mesma função (`processar_retina`) é usada tanto pela inspeção visual
quanto pelo processamento em lote.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import cv2
import matplotlib.pyplot as plt
import numpy as np
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


def centralizar_retina(img, mascara, tamanho_imagem):
    """
    Recorta um quadrado centrado na retina, com lado igual ao diâmetro
    do círculo, e redimensiona para `tamanho_imagem`.

    Em muitas fotos do EyePACS a câmera corta o topo e a base do círculo
    da retina, então o raio vem da maior dimensão da caixa delimitadora
    da máscara. O que ficar fora da foto original é preenchido com preto.

    Retorna:
        (imagem_quadrada, status)
    """

    # Localiza os pixels pertencentes à região detectada
    pontos = cv2.findNonZero(mascara)

    # Se nenhum ponto for encontrado, registra a falha
    if pontos is None:
        return None, "mascara_vazia"

    # Calcula a caixa delimitadora da retina
    x, y, w, h = cv2.boundingRect(pontos)

    raio = max(w, h) // 2

    if raio < configuracao.RAIO_MINIMO:
        return None, "recorte_invalido"

    centro_x = x + w // 2
    centro_y = y + h // 2

    altura, largura = img.shape[:2]

    # Borda necessária para que o quadrado caiba inteiro na imagem
    borda = max(
        0,
        raio - centro_x,
        raio - centro_y,
        centro_x + raio - largura,
        centro_y + raio - altura
    )

    if borda > 0:
        img = cv2.copyMakeBorder(
            img,
            borda, borda, borda, borda,
            cv2.BORDER_CONSTANT,
            value=0
        )

        centro_x += borda
        centro_y += borda

    quadrado = img[
        centro_y - raio:centro_y + raio,
        centro_x - raio:centro_x + raio
    ]

    if quadrado.size == 0:
        return None, "recorte_invalido"

    # INTER_AREA evita serrilhado ao reduzir fotos de ~3000 pixels
    redimensionada = cv2.resize(
        quadrado,
        tamanho_imagem,
        interpolation=cv2.INTER_AREA
    )

    return redimensionada, "ok"


def normalizar_cor_ben(img, sigma=configuracao.SIGMA_BEN):
    """
    Normalização de cor de Ben Graham: subtrai a cor média local
    (desfoque gaussiano) e centraliza em 128.

    Remove variações de iluminação e de câmera e realça estruturas
    pequenas como microaneurismas e exsudatos.
    """

    desfocada = cv2.GaussianBlur(
        img,
        (0, 0),
        sigma
    )

    return cv2.addWeighted(
        img, 4,
        desfocada, -4,
        128
    )


def aplicar_mascara_circular(
    img,
    fracao_raio=configuracao.RAIO_MASCARA,
    cor_fundo=configuracao.COR_FUNDO
):
    """
    Mantém apenas o disco central da imagem (já centralizada na retina)
    e pinta o restante com a cor de fundo, descartando a borda do
    círculo, onde ficam os artefatos de iluminação.
    """

    altura, largura = img.shape[:2]

    mascara = np.zeros(
        (altura, largura),
        dtype=np.uint8
    )

    cv2.circle(
        mascara,
        (largura // 2, altura // 2),
        int(min(altura, largura) / 2 * fracao_raio),
        255,
        -1
    )

    resultado = np.full_like(img, cor_fundo)
    resultado[mascara > 0] = img[mascara > 0]

    return resultado


def processar_retina(
    img,
    tamanho_imagem=configuracao.TAMANHO_IMAGEM,
    aplicar_ben=configuracao.APLICAR_BEN
):
    """
    Pré-processamento completo de uma foto de fundo de olho:
    máscara de Otsu, recorte quadrado centrado na retina, normalização
    de cor de Ben Graham (opcional) e máscara circular.

    Retorna:
        (imagem_processada, mascara_otsu, status)

    Onde status pode ser:
        "ok"               -> processamento válido
        "mascara_vazia"    -> Otsu não encontrou nenhum pixel
        "recorte_invalido" -> a retina encontrada é pequena demais

    Quando o status é diferente de "ok", a imagem devolvida é None.
    """

    mascara = criar_mascara_otsu(img)

    quadrada, status = centralizar_retina(
        img,
        mascara,
        tamanho_imagem
    )

    if status != "ok":
        return None, mascara, status

    if aplicar_ben:
        quadrada = normalizar_cor_ben(quadrada)

    processada = aplicar_mascara_circular(quadrada)

    return processada, mascara, "ok"


def pre_visualizar_recorte(
    dados,
    pasta_imagens=configuracao.PASTA_IMAGENS,
    numero_amostras=configuracao.NUMERO_AMOSTRAS_INSPECAO,
    seed=configuracao.SEED
):
    """
    Inspeção visual: mostra, para uma amostra de imagens, a foto
    original, a máscara de Otsu e a imagem processada.
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

        processada, thresh, status = processar_retina(img)

        if status == "mascara_vazia":
            print(f"Máscara vazia para a imagem: {nome}")
            processada = img

        elif status != "ok":
            print(f"Recorte inválido para a imagem: {nome}")
            processada = img

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
        plt.imshow(cv2.cvtColor(processada, cv2.COLOR_BGR2RGB))
        plt.title("Processada")
        plt.axis("off")

        plt.tight_layout()
        plt.show()


def processar_e_salvar_imagem(
    img_nome,
    pasta_imagens,
    pasta_saida,
    tamanho_imagem,
    qualidade_jpeg=configuracao.QUALIDADE_JPEG
):
    """
    Lê, processa e salva uma imagem.

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

    img_final, _, status = processar_retina(
        img,
        tamanho_imagem=tamanho_imagem
    )

    if status != "ok":
        return img_nome, status

    # Salva a imagem e guarda o resultado da operação
    sucesso = cv2.imwrite(
        caminho_out,
        img_final,
        [cv2.IMWRITE_JPEG_QUALITY, qualidade_jpeg]
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
    Processa todas as imagens do dataset em paralelo,
    salvando o resultado em `pasta_saida`.

    A execução é interrompida com RuntimeError se alguma imagem falhar
    ou se alguma imagem esperada não estiver na pasta final.

    Retorna o DataFrame com o status de cada imagem.
    """

    os.makedirs(
        pasta_saida,
        exist_ok=True
    )

    print("Iniciando o pré-processamento em paralelo...")

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
