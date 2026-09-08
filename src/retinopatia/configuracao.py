"""
Configurações centrais do projeto.

Todos os valores que antes estavam espalhados como literais pelas
células do notebook ficam reunidos aqui. Os valores são exatamente os
mesmos utilizados na versão original do notebook.

Atenção: qualquer mudança no pré-processamento, na divisão treino/
validação ou no mapeamento binário dos rótulos invalida os arquivos já
gerados em PASTA_IMAGENS_OTIMIZADAS e os CSVs de split salvos no Drive.
Nesses casos, regenere os caches ou mude a versão no caminho.
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

# Pasta desta versão do pré-processamento
PASTA_IMAGENS_OTIMIZADAS = "/content/dataset_otimizado_otsu_v1"

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
# DIVISÃO TREINO / VALIDAÇÃO
# ============================================================

NUMERO_SPLITS = 5

# ============================================================
# PRÉ-PROCESSAMENTO
# ============================================================

TAMANHO_IMAGEM = (224, 224)

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
# GERADORES DE DADOS
# ============================================================

# Como temos muita memória de GPU disponível, o lote é grande para
# acelerar o treino. Se houver erro de falta de memória (OOM),
# diminua para 64 ou 32.
TAMANHO_LOTE = 128

# Data augmentation aplicado somente ao conjunto de treino
AUMENTO_DADOS = {
    "rotation_range": 20,       # Roda a imagem até 20 graus
    "horizontal_flip": True,    # Espelhamento horizontal
    "vertical_flip": True,      # Espelhamento vertical
    "zoom_range": 0.1,          # Aplica um leve zoom aleatório
    "fill_mode": "constant",    # Preenche as margens com preto
    "cval": 0
}

# ============================================================
# MODELO
# ============================================================

FORMATO_ENTRADA = TAMANHO_IMAGEM + (3,)

UNIDADES_DENSA = 256

TAXA_DROPOUT = 0.5

# ============================================================
# CALLBACKS (comuns às duas fases)
# ============================================================

METRICA_MONITORADA = "val_pr_auc"

FATOR_REDUCE_LR = 0.2

# ============================================================
# TREINO — FASE 1 (apenas a cabeça da rede)
# ============================================================

EPOCAS_FASE1 = 15

TAXA_APRENDIZADO_FASE1 = 1e-3

CAMINHO_PESOS_FASE1 = "melhor_modelo_retinopatia.weights.h5"

PACIENCIA_EARLY_STOP_FASE1 = 6

PACIENCIA_REDUCE_LR_FASE1 = 3

LR_MINIMO_FASE1 = 1e-6

# ============================================================
# TREINO — FASE 2 (fine-tuning)
# ============================================================

EPOCAS_FASE2 = 15

TAXA_APRENDIZADO_FASE2 = 1e-5

CAMINHO_PESOS_FASE2 = "melhor_modelo_retinopatia_fase2.weights.h5"

PACIENCIA_EARLY_STOP_FASE2 = 8

PACIENCIA_REDUCE_LR_FASE2 = 3

LR_MINIMO_FASE2 = 1e-7

# ============================================================
# AVALIAÇÃO
# ============================================================

# Contexto de rastreio médico: falsos negativos custam mais caro que
# falsos positivos, por isso o limiar é escolhido pela sensibilidade.
SENSIBILIDADE_DESEJADA = 0.90

# ============================================================
# MODELO FINAL
# ============================================================

CAMINHO_MODELO_FINAL = "modelo_retinopatia_final.keras"
