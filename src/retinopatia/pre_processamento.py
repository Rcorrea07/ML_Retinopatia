"""
Pré-processamento das imagens (versão 3).

Segue a receita das soluções vencedoras das competições do Kaggle de
retinopatia (Ben Graham em 2015 e os notebooks de topo da APTOS 2019):

1. Uma máscara localiza a retina e descarta a moldura preta.
2. Um quadrado centrado na retina, com lado igual ao diâmetro do
   círculo, é recortado e redimensionado. Assim toda retina ocupa a
   mesma área da imagem, independentemente da câmera.
3. A normalização de cor de Ben Graham remove diferenças de iluminação.
4. Uma máscara circular descarta a borda do círculo.

O que mudou da v2 para a v3 (as duas continuam disponíveis pelos
parâmetros `metodo_mascara` e `ben_respeita_mascara`):

- A máscara usa um limiar fixo baixo em vez de Otsu, que falhava em
  fotos escuras e pegava só parte da retina.
- A normalização de Ben calcula a média local só dentro da retina. Na
  v2, onde a câmera corta o topo e a base do círculo, o desfoque
  misturava a retina com o preto do preenchimento e criava faixas
  claras e escuras que a máscara circular não removia.
- Imagens cuja máscara cobre pouco da área esperada da retina são
  salvas, mas marcadas como "mascara_suspeita" para inspeção.

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


def criar_mascara_retina(
    img,
    metodo_mascara=configuracao.METODO_MASCARA,
    limiar=configuracao.LIMIAR_MASCARA
):
    """
    Cria a máscara binária (0 ou 255) da retina.

    Com "limiar_fixo", mantém os pixels acima de `limiar`, fica só com a
    maior região conexa e preenche a sua envoltória convexa, o que fecha
    buracos (como a fóvea escura) e reentrâncias. Em fotos quase
    totalmente pretas o limiar fixo não acha uma retina de tamanho
    plausível; nesses casos a máscara volta a ser a de Otsu. Com "otsu",
    reproduz a máscara da v2.
    """

    if metodo_mascara not in ("limiar_fixo", "otsu"):
        raise ValueError(
            f"Método de máscara desconhecido: {metodo_mascara}"
        )

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

    _, mascara_otsu = cv2.threshold(
        gray_blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    if metodo_mascara == "otsu":
        return mascara_otsu

    _, mascara = cv2.threshold(
        gray_blur,
        limiar,
        255,
        cv2.THRESH_BINARY
    )

    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contornos:
        return mascara_otsu

    maior_contorno = max(contornos, key=cv2.contourArea)

    _, _, largura, altura = cv2.boundingRect(maior_contorno)

    if max(largura, altura) < (
        configuracao.FRACAO_MINIMA_DIAMETRO_RETINA * min(gray.shape)
    ):
        return mascara_otsu

    mascara = np.zeros_like(mascara)

    cv2.fillConvexPoly(
        mascara,
        cv2.convexHull(maior_contorno),
        255
    )

    return mascara


def _recortar_quadrado(img, centro_x, centro_y, raio, borda):
    """
    Completa a imagem com zeros em volta (se `borda` > 0) e recorta o
    quadrado de lado 2 * raio centrado em (centro_x, centro_y).
    """

    if borda > 0:
        img = cv2.copyMakeBorder(
            img,
            borda, borda, borda, borda,
            cv2.BORDER_CONSTANT,
            value=0
        )

    centro_x += borda
    centro_y += borda

    return img[
        centro_y - raio:centro_y + raio,
        centro_x - raio:centro_x + raio
    ]


def centralizar_retina(img, mascara, tamanho_imagem):
    """
    Recorta um quadrado centrado na retina, com lado igual ao diâmetro
    do círculo, e redimensiona para `tamanho_imagem`.

    Em muitas fotos do EyePACS a câmera corta o topo e a base do círculo
    da retina, então o raio vem da maior dimensão da caixa delimitadora
    da máscara. O que ficar fora da foto original é preenchido com preto.

    O mesmo recorte é aplicado à máscara da retina e a uma máscara da
    área coberta pela foto original, usadas na normalização de Ben e na
    verificação da máscara.

    Retorna:
        (imagem_quadrada, mascara_quadrada, area_foto_quadrada, status)
    """

    # Localiza os pixels pertencentes à região detectada
    pontos = cv2.findNonZero(mascara)

    # Se nenhum ponto for encontrado, registra a falha
    if pontos is None:
        return None, None, None, "mascara_vazia"

    # Calcula a caixa delimitadora da retina
    x, y, w, h = cv2.boundingRect(pontos)

    raio = max(w, h) // 2

    if raio < configuracao.RAIO_MINIMO:
        return None, None, None, "recorte_invalido"

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

    area_foto = np.full(
        (altura, largura),
        255,
        dtype=np.uint8
    )

    recortes = [
        _recortar_quadrado(camada, centro_x, centro_y, raio, borda)
        for camada in (img, mascara, area_foto)
    ]

    if recortes[0].size == 0:
        return None, None, None, "recorte_invalido"

    # INTER_AREA evita serrilhado ao reduzir fotos de ~3000 pixels;
    # as máscaras usam o vizinho mais próximo para continuarem binárias
    interpolacoes = (
        cv2.INTER_AREA,
        cv2.INTER_NEAREST,
        cv2.INTER_NEAREST
    )

    imagem_quadrada, mascara_quadrada, area_foto_quadrada = (
        cv2.resize(recorte, tamanho_imagem, interpolation=interpolacao)
        for recorte, interpolacao in zip(recortes, interpolacoes)
    )

    return imagem_quadrada, mascara_quadrada, area_foto_quadrada, "ok"


def fracao_mascara(mascara_quadrada, area_foto_quadrada):
    """
    Fração da área esperada da retina coberta pela máscara.

    A área esperada é o disco inscrito no quadrado, limitado à parte
    coberta pela foto original (a câmera às vezes corta o topo e a base
    do círculo). Uma máscara boa fica perto de 1; uma que pegou só parte
    da retina, como o Otsu em fotos escuras, fica bem abaixo.
    """

    altura, largura = mascara_quadrada.shape[:2]

    disco = np.zeros(
        (altura, largura),
        dtype=np.uint8
    )

    cv2.circle(
        disco,
        (largura // 2, altura // 2),
        min(altura, largura) // 2,
        255,
        -1
    )

    esperada = (disco > 0) & (area_foto_quadrada > 0)

    coberta = esperada & (mascara_quadrada > 0)

    return coberta.sum() / max(esperada.sum(), 1)


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


def normalizar_cor_ben_na_retina(
    img,
    mascara,
    sigma=configuracao.SIGMA_BEN,
    erosao=configuracao.EROSAO_MASCARA,
    cor_fundo=configuracao.COR_FUNDO
):
    """
    Normalização de Ben Graham calculada só dentro da retina.

    A média local é um desfoque gaussiano normalizado pela máscara
    (desfoque(imagem * máscara) / desfoque(máscara)), então o preto fora
    da retina não entra na conta e não aparecem faixas na fronteira.
    Fora da retina (já com a erosão), a imagem recebe `cor_fundo`.
    """

    retina = (mascara > 0).astype(np.uint8)

    if erosao > 0:
        retina = cv2.erode(
            retina,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * erosao + 1, 2 * erosao + 1)
            )
        )

    peso = retina.astype(np.float32)
    imagem = img.astype(np.float32)

    soma_local = cv2.GaussianBlur(
        imagem * peso[..., None],
        (0, 0),
        sigma
    )

    peso_local = cv2.GaussianBlur(
        peso,
        (0, 0),
        sigma
    )

    media_local = soma_local / np.maximum(peso_local, 1e-3)[..., None]

    normalizada = np.clip(
        4 * imagem - 4 * media_local + 128,
        0,
        255
    )

    return np.where(
        retina[..., None] > 0,
        normalizada,
        cor_fundo
    ).astype(np.uint8)


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
    aplicar_ben=configuracao.APLICAR_BEN,
    metodo_mascara=configuracao.METODO_MASCARA,
    ben_respeita_mascara=configuracao.BEN_RESPEITA_MASCARA
):
    """
    Pré-processamento completo de uma foto de fundo de olho:
    máscara da retina, recorte quadrado centrado na retina, normalização
    de cor de Ben Graham (opcional) e máscara circular.

    Retorna:
        (imagem_processada, mascara_retina, status)

    Onde status pode ser:
        "ok"               -> processamento válido
        "mascara_suspeita" -> processada, mas a máscara cobre menos de
                              FRACAO_MINIMA_MASCARA da retina esperada
        "mascara_vazia"    -> a máscara não encontrou nenhum pixel
        "recorte_invalido" -> a retina encontrada é pequena demais

    Nos dois últimos casos a imagem devolvida é None.
    """

    mascara = criar_mascara_retina(img, metodo_mascara)

    quadrada, mascara_quadrada, area_foto, status = centralizar_retina(
        img,
        mascara,
        tamanho_imagem
    )

    if status != "ok":
        return None, mascara, status

    if aplicar_ben and ben_respeita_mascara:
        quadrada = normalizar_cor_ben_na_retina(quadrada, mascara_quadrada)

    elif aplicar_ben:
        quadrada = normalizar_cor_ben(quadrada)

    processada = aplicar_mascara_circular(quadrada)

    if (
        fracao_mascara(mascara_quadrada, area_foto)
        < configuracao.FRACAO_MINIMA_MASCARA
    ):
        status = configuracao.STATUS_SUSPEITO

    return processada, mascara, status


def pre_visualizar_recorte(
    dados,
    imagens=None,
    pasta_imagens=configuracao.PASTA_IMAGENS,
    numero_amostras=configuracao.NUMERO_AMOSTRAS_INSPECAO,
    seed=configuracao.SEED
):
    """
    Inspeção visual: mostra a foto original, a máscara da retina e a
    imagem processada.

    Sem `imagens`, usa uma amostra aleatória de `dados`; com uma lista
    de nomes (por exemplo, as máscaras suspeitas), mostra até
    `numero_amostras` delas.

    Precisa das fotos originais: quando as imagens processadas vieram do
    cache no Drive, o dataset bruto não foi extraído e a inspeção é
    pulada.
    """

    if not os.path.isdir(pasta_imagens):
        print(
            "As fotos originais não foram extraídas nesta sessão "
            "(as imagens processadas vieram do cache). "
            "Inspeção visual pulada."
        )
        return

    if imagens is None:
        nomes = dados.sample(
            n=min(numero_amostras, len(dados)),
            random_state=seed
        )["image"]

    else:
        nomes = list(imagens)[:numero_amostras]

    for nome in nomes:
        caminho = os.path.join(pasta_imagens, nome)
        img = cv2.imread(caminho)

        if img is None:
            print(f"Erro ao ler: {nome}")
            continue

        processada, thresh, status = processar_retina(img)

        if status == "mascara_vazia":
            print(f"Máscara vazia para a imagem: {nome}")
            processada = img

        elif status == "recorte_invalido":
            print(f"Recorte inválido para a imagem: {nome}")
            processada = img

        elif status == configuracao.STATUS_SUSPEITO:
            print(f"Máscara suspeita para a imagem: {nome}")

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(thresh, cmap="gray")
        plt.title("Máscara da retina")
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
    qualidade_jpeg=configuracao.QUALIDADE_JPEG,
    **opcoes_processamento
):
    """
    Lê, processa e salva uma imagem. As `opcoes_processamento` são
    repassadas a `processar_retina` (por exemplo, para reproduzir a v2).

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
        tamanho_imagem=tamanho_imagem,
        **opcoes_processamento
    )

    if img_final is None:
        return img_nome, status

    # Salva a imagem e guarda o resultado da operação
    sucesso = cv2.imwrite(
        caminho_out,
        img_final,
        [cv2.IMWRITE_JPEG_QUALITY, qualidade_jpeg]
    )

    if not sucesso:
        return img_nome, "erro_escrita"

    if status == configuracao.STATUS_SUSPEITO:
        return img_nome, status

    return img_nome, "processada"


def pre_processar_dataset(
    dados,
    pasta_imagens=configuracao.PASTA_IMAGENS,
    pasta_saida=configuracao.PASTA_IMAGENS_OTIMIZADAS,
    tamanho_imagem=configuracao.TAMANHO_IMAGEM,
    maximo_workers=configuracao.MAXIMO_WORKERS,
    caminho_lista_suspeitas=configuracao.CAMINHO_LISTA_SUSPEITAS,
    **opcoes_processamento
):
    """
    Processa todas as imagens do dataset em paralelo,
    salvando o resultado em `pasta_saida`.

    A execução é interrompida com RuntimeError se alguma imagem falhar
    ou se alguma imagem esperada não estiver na pasta final. As imagens
    com máscara suspeita não interrompem: a lista delas é salva em
    `caminho_lista_suspeitas` (se não for None).

    As `opcoes_processamento` são repassadas a `processar_retina`.

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
        tamanho_imagem=tamanho_imagem,
        **opcoes_processamento
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

    # A lista das suspeitas é salva antes da checagem de falhas: numa
    # nova execução as imagens já salvas viram "existente" e a lista
    # não poderia mais ser refeita
    suspeitas = resultados_df[
        resultados_df["status"] == configuracao.STATUS_SUSPEITO
    ]

    if not suspeitas.empty:
        print(
            f"\n{len(suspeitas)} imagens ({len(suspeitas) / len(dados):.2%}) "
            "foram salvas com máscara suspeita. Veja-as com "
            "pre_visualizar_recorte(dados, imagens=...)."
        )

        if caminho_lista_suspeitas is not None:
            os.makedirs(
                os.path.dirname(caminho_lista_suspeitas),
                exist_ok=True
            )

            suspeitas.to_csv(caminho_lista_suspeitas, index=False)

            print(f"Lista salva em: {caminho_lista_suspeitas}")

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
