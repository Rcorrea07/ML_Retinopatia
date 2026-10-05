"""
Configurações centrais do projeto.

Todos os hiperparâmetros, caminhos e constantes ficam reunidos aqui. Os
valores da versão 2 seguem o que funcionou nas melhores soluções das
competições do Kaggle de retinopatia diabética (EyePACS 2015 e APTOS
2019): imagens grandes, normalização de cor de Ben Graham, máscara
circular, GeM pooling e uma cabeça auxiliar que aprende o grau 0 a 4.

Atenção: qualquer mudança no pré-processamento, na divisão treino/
validação/teste ou no mapeamento dos rótulos invalida os arquivos já
gerados em PASTA_IMAGENS_OTIMIZADAS e os CSVs de split salvos no Drive.
Nesses casos, regenere os caches ou mude a versão no caminho
(VERSAO_PRE_PROCESSAMENTO / VERSAO_SPLIT).
"""

import os

# ============================================================
# REPRODUTIBILIDADE
# ============================================================

SEED = 42

# ============================================================
# CAMINHOS (ambiente do Google Colab)
# ============================================================

CAMINHO_ZIP_PRINCIPAL = (
    "/content/drive/MyDrive/Datasets/"
    "diabetic-retinopathy-detection.zip"
)

PASTA_BASE = "/content/dataset_retinopatia"

PASTA_IMAGENS = os.path.join(
    PASTA_BASE,
    "train"
)

CAMINHO_CSV = os.path.join(
    PASTA_BASE,
    "trainLabels.csv"
)

PASTA_SPLITS = (
    "/content/drive/MyDrive/"
    "ProjetoRetinopatia/splits"
)

# ============================================================
# RÓTULOS
# ============================================================

# Graus de severidade aceitos no CSV original
NIVEIS_PERMITIDOS = {0, 1, 2, 3, 4}

# Nomes das classes binárias usadas pelo modelo
NOMES_CLASSES = {
    0: "Sem retinopatia",
    1: "Com retinopatia, graus originais 1 a 4"
}

# Rótulos curtos usados nos relatórios e gráficos
ROTULOS_RELATORIO = [
    "Saudável",
    "Doente"
]

ROTULOS_MATRIZ = [
    "0 (Saudável)",
    "1 (Doente)"
]

# ============================================================
# DIVISÃO TREINO / VALIDAÇÃO / TESTE
# ============================================================

# Com 10 dobras, cada dobra tem cerca de 10% dos pacientes:
# uma vira teste, outra validação e as outras 8 ficam no treino.
NUMERO_SPLITS = 10

DOBRA_TESTE = 0

DOBRA_VALIDACAO = 1

# Versão do esquema de divisão. A v1 tinha apenas treino e validação;
# a v2 separa também um conjunto de teste que nunca é usado para
# escolher limiar nem checkpoint.
VERSAO_SPLIT = "v2"

# ============================================================
# PRÉ-PROCESSAMENTO
# ============================================================

# As soluções vencedoras usaram de 384 a 512 pixels: lesões pequenas,
# como microaneurismas (grau 1), somem em 224 pixels.
TAMANHO_IMAGEM = (448, 448)

# Normalização de cor de Ben Graham (vencedor de 2015):
# 4 * imagem - 4 * desfoque_gaussiano(imagem) + 128.
# Remove as diferenças de iluminação entre câmeras e realça lesões.
APLICAR_BEN = True

SIGMA_BEN = 10

# Fração do raio da retina mantida pela máscara circular. A borda do
# círculo costuma ter artefatos de iluminação, por isso é descartada.
RAIO_MASCARA = 0.9

# Cor do fundo fora da máscara. Com a normalização de Ben, o cinza 128
# é a cor "neutra"; sem ela, o fundo é preto.
COR_FUNDO = 128 if APLICAR_BEN else 0

# Raio mínimo, em pixels, para considerar que a retina foi encontrada
RAIO_MINIMO = 10

QUALIDADE_JPEG = 95

VERSAO_PRE_PROCESSAMENTO = "v2"

# Pasta desta versão do pré-processamento
PASTA_IMAGENS_OTIMIZADAS = (
    f"/content/dataset_retina_{VERSAO_PRE_PROCESSAMENTO}_"
    f"{TAMANHO_IMAGEM[0]}"
)

# Limita o número de threads para evitar sobrecarga
MAXIMO_WORKERS = 8

# Quantidade de imagens mostradas na inspeção visual
NUMERO_AMOSTRAS_INSPECAO = 20

# Estados que representam falha no pré-processamento
STATUS_DE_ERRO = [
    "erro_leitura",
    "mascara_vazia",
    "recorte_invalido",
    "erro_escrita"
]

# ============================================================
# PIPELINE DE DADOS
# ============================================================

# Valor inicial para EfficientNetB4 em 448 pixels numa GPU L4 com
# precisão mista. Se houver erro de falta de memória (OOM), diminua
# para 16; numa A100 dá para subir para 32.
TAMANHO_LOTE = 24

# Data augmentation aplicado somente ao conjunto de treino.
# A retina não tem orientação canônica, então rotações de 360 graus
# e espelhamentos são seguros.
AUMENTO_DADOS = {
    "fator_rotacao": 0.5,       # Fração de volta completa: 0.5 = ±180 graus
    "fator_zoom": 0.15,         # Zoom aleatório de ±15%
    "fator_brilho": 0.1,        # Brilho de ±10% da faixa 0 a 255
    "fator_contraste": 0.1      # Contraste entre 0.9 e 1.1
}

# ============================================================
# MODELO
# ============================================================

FORMATO_ENTRADA = TAMANHO_IMAGEM + (3,)

# Opções: "EfficientNetB3", "EfficientNetB4", "EfficientNetB5",
# "EfficientNetV2S". Todas esperam a entrada na faixa de 0 a 255.
BACKBONE = "EfficientNetB4"

# Precisão mista (float16 nos cálculos, float32 nos pesos): quase
# dobra a velocidade em GPUs L4/A100/T4 e reduz o uso de memória.
PRECISAO_MISTA = True

# Generalized Mean pooling (1º lugar APTOS 2019): p = 1 equivale à
# média, p grande se aproxima do máximo. O p é aprendido no treino.
GEM_P_INICIAL = 3.0

TAXA_DROPOUT = 0.4

# Nomes das duas saídas do modelo
SAIDA_DOENTE = "doente"     # Decisão binária (sigmoide)
SAIDA_GRAU = "grau"         # Grau 0 a 4 (regressão, cabeça auxiliar)

# Peso da perda da cabeça de grau na perda total. A cabeça de grau
# existe para que a rede aprenda com a informação ordinal dos cinco
# graus; a decisão continua sendo tomada pela cabeça binária.
PESO_PERDA_GRAU = 0.5

# Perda de Huber (SmoothL1) na regressão do grau: menos sensível a
# rótulos errados do que o erro quadrático.
DELTA_HUBER = 1.0

# ============================================================
# CALLBACKS (comuns às duas fases)
# ============================================================

METRICA_MONITORADA = f"val_{SAIDA_DOENTE}_pr_auc"

FATOR_REDUCE_LR = 0.2

# ============================================================
# TREINO — FASE 1 (apenas as cabeças da rede)
# ============================================================

EPOCAS_FASE1 = 8

TAXA_APRENDIZADO_FASE1 = 1e-3

CAMINHO_PESOS_FASE1 = "melhor_modelo_retinopatia.weights.h5"

PACIENCIA_EARLY_STOP_FASE1 = 3

PACIENCIA_REDUCE_LR_FASE1 = 2

LR_MINIMO_FASE1 = 1e-6

# ============================================================
# TREINO — FASE 2 (fine-tuning)
# ============================================================

EPOCAS_FASE2 = 20

TAXA_APRENDIZADO_FASE2 = 5e-5

CAMINHO_PESOS_FASE2 = "melhor_modelo_retinopatia_fase2.weights.h5"

PACIENCIA_EARLY_STOP_FASE2 = 6

PACIENCIA_REDUCE_LR_FASE2 = 2

LR_MINIMO_FASE2 = 1e-7

# ============================================================
# AVALIAÇÃO
# ============================================================

# Contexto de rastreio médico: falsos negativos custam mais caro que
# falsos positivos, por isso o limiar é escolhido pela sensibilidade.
SENSIBILIDADE_DESEJADA = 0.90

# Test-time augmentation: média das predições da imagem original e de
# seus espelhamentos horizontal, vertical e em ambos os eixos.
TTA_ATIVO = True

# ============================================================
# MODELO FINAL
# ============================================================

CAMINHO_MODELO_FINAL = "modelo_retinopatia_final.keras"
