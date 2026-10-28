"""
Experimentos nomeados do projeto.

Cada experimento é um dicionário de constantes de `configuracao` que
mudam em relação ao padrão. O notebook escolhe um na primeira célula
(variável de ambiente RETINOPATIA_EXPERIMENTO), antes de importar o
pacote; `configuracao` aplica as mudanças antes de calcular os caminhos
derivados (pasta de resultados, cache, nome do modelo).

Para rodar experimentos em paralelo, cada sessão do Colab usa uma cópia
do mesmo notebook, mudando só o nome do experimento.

Este módulo não importa TensorFlow: pode ser lido e testado fora do
Colab.

Referência de comparação (campanha de 20/10/2026, E01: B0 448, v3,
dados extras): validação por paciente, ROC-AUC 0,908 e especificidade
0,671 com sensibilidade 0,90; com TTA de 4 variantes, 0,913 e 0,711.
As fontes de cada ideia estão em docs/REFERENCIAS.md.
"""

import os

# Nome usado quando a variável de ambiente não é definida: o padrão de
# `configuracao`, sem nenhuma mudança
EXPERIMENTO_PADRAO = "padrao"

EXPERIMENTOS = {
    EXPERIMENTO_PADRAO: {},

    # --------------------------------------------------------
    # Rodada 1: resolução e destilação, cada uma isolada
    # --------------------------------------------------------

    # A campanha mostrou que baixar de 448 para 384 custou 13 pontos de
    # especificidade; as soluções de topo de 2015 usaram 512 a 768.
    # Fontes: o_O (redes de 512) e Reformed Gamblers (385 contra 767).
    "R01_b0_512": {
        "BACKBONE": "EfficientNetB0",
        "TAMANHO_IMAGEM": (512, 512),
    },

    "R02_b0_640": {
        "BACKBONE": "EfficientNetB0",
        "TAMANHO_IMAGEM": (640, 640),
    },

    # A B4 448 com dados extras (melhor modelo único até agora) ensina a
    # B0 na mesma resolução, para medir só o efeito da destilação.
    # Fonte: exemplo oficial de destilação do Keras.
    "R03_b0_448_destilacao": {
        "BACKBONE": "EfficientNetB0",
        "DESTILACAO": True,
    },

    # --------------------------------------------------------
    # Rodada 2, em 640 px: o R02 (B0 640) venceu a rodada 1 com
    # especificidade 0,732 por paciente (E01: 0,671)
    # --------------------------------------------------------

    # Os dois olhos do paciente na mesma decisão, com pesos
    # compartilhados. Fontes: Jeffrey De Fauw e o_O (blend por paciente).
    # Custa cerca do dobro de um treino comum: só roda se o combinador
    # dos dois olhos (analise.combinador_paciente) mostrar ganho.
    "R04_b0_pares_640": {
        "BACKBONE": "EfficientNetB0",
        "TAMANHO_IMAGEM": (640, 640),
        "MODELO_PARES": True,
    },

    # Fonte: 1º lugar APTOS 2019 (EfficientNets, GeM, regressão).
    "R05_v2b0_640": {
        "BACKBONE": "EfficientNetV2B0",
        "TAMANHO_IMAGEM": (640, 640),
    },

    # O R02 teve a melhor época na última (12 de 12): ainda melhorava.
    # Mesma receita, com 18 épocas na Fase 2 (o cosseno se estica junto).
    "R06_b0_640_longo": {
        "BACKBONE": "EfficientNetB0",
        "TAMANHO_IMAGEM": (640, 640),
        "EPOCAS_FASE2": 18,
        "PACIENCIA_EARLY_STOP_FASE2": 4,
    },

    # --------------------------------------------------------
    # Rodada 3: R02 e R05 com a semente 17 (notebooks R07 e R08, com
    # RETINOPATIA_SEED_TREINO) e a reserva em 768 px
    # --------------------------------------------------------

    # O grau 1 (só microaneurismas) ficou em ~70% em todos os treinos:
    # 448 → 640 cortou os falsos positivos no grau 0, mas não mexeu nele.
    # Mesma receita do R02 em 768 px, para medir só a resolução.
    # Fonte: Reformed Gamblers (385 contra 767 px).
    "R10_b0_768": {
        "BACKBONE": "EfficientNetB0",
        "TAMANHO_IMAGEM": (768, 768),
    },

    # --------------------------------------------------------
    # Rodada 4: as duas alavancas da rodada 3 juntas. A V2-B0 venceu a
    # B0 em 640 px nas duas sementes (por paciente), e o B0 768 foi o
    # melhor por olho. Roda com o R10 semente 17 numa sessão só
    # (execucao.rodar_rodada), os três em 768 px.
    # --------------------------------------------------------

    "R11_v2b0_768": {
        "BACKBONE": "EfficientNetV2B0",
        "TAMANHO_IMAGEM": (768, 768),
    },
}


def nome_experimento_ativo():
    """
    Nome do experimento escolhido na variável de ambiente
    RETINOPATIA_EXPERIMENTO (o padrão quando ela não existe).
    """

    return os.environ.get(
        "RETINOPATIA_EXPERIMENTO",
        EXPERIMENTO_PADRAO
    ).strip() or EXPERIMENTO_PADRAO


def sobreposicoes(nome=None):
    """
    Devolve uma cópia das constantes que o experimento muda. Um nome
    desconhecido para a execução com a lista dos nomes válidos.
    """

    nome = nome or nome_experimento_ativo()

    if nome not in EXPERIMENTOS:
        raise ValueError(
            f"Experimento desconhecido: {nome}. "
            f"Opções: {sorted(EXPERIMENTOS)}"
        )

    return dict(EXPERIMENTOS[nome])
