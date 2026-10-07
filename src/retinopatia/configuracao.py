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
import subprocess

# ============================================================
# REPRODUTIBILIDADE
# ============================================================

SEED = 42

# Operações determinísticas no TensorFlow. Garantem o mesmo resultado a
# cada execução, mas podem deixar o treino mais lento ou falhar em
# operações sem implementação determinística na GPU. Se
# `pipeline_dados.medir_velocidade` ou o tempo por época mostrarem
# gargalo, ou se aparecer erro de determinismo, desligue.
OPERACOES_DETERMINISTICAS = True

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

# Pasta do projeto no Drive: tudo o que precisa sobreviver a uma
# desconexão do Colab fica aqui (o disco /content é apagado).
# Organização:
#   modelos/          modelos finais (.keras), um por treino
#   resultados/       uma pasta por treino (pesos, históricos, predições,
#                     configuração) e o registro de todos os treinos
#   relatorios/       relatórios em PDF
#   notebooks/        cópias dos notebooks executados, com as saídas
#   dados_internos/   arquivos de trabalho do pipeline (splits e cache)
PASTA_PROJETO_DRIVE = "/content/drive/MyDrive/ProjetoRetinopatia"

PASTA_MODELOS = os.path.join(
    PASTA_PROJETO_DRIVE,
    "modelos"
)

PASTA_DADOS_INTERNOS = os.path.join(
    PASTA_PROJETO_DRIVE,
    "dados_internos"
)

# Divisão treino/validação/teste salva uma vez e reutilizada: garante
# que todos os treinos sejam comparados nos mesmos pacientes
PASTA_SPLITS = os.path.join(
    PASTA_DADOS_INTERNOS,
    "splits"
)

# Zips das imagens já processadas, um por versão do pré-processamento
PASTA_CACHE_DRIVE = os.path.join(
    PASTA_DADOS_INTERNOS,
    "cache"
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

# Como a máscara da retina é criada:
# - "limiar_fixo" (v3): pixels acima de LIMIAR_MASCARA, maior região
#   conexa e envoltória convexa. Funciona também em fotos escuras.
# - "otsu" (v2): limiar de Otsu. Em fotos escuras separa "retina escura"
#   de "retina menos escura" e pega só parte do círculo.
METODO_MASCARA = "limiar_fixo"

# Limiar, em tons de cinza de 0 a 255, que separa a retina do fundo preto
LIMIAR_MASCARA = 10

# Se a região achada pelo limiar fixo tiver diâmetro menor que esta
# fração do menor lado da foto, a foto é escura demais para o limiar
# fixo e a máscara volta a ser a de Otsu (que processou essas fotos na
# v2). Na primeira execução da v3, 4 das 35.126 fotos caíram nesse caso.
FRACAO_MINIMA_DIAMETRO_RETINA = 0.25

# Com True (v3), a média local da normalização de Ben é calculada só
# dentro da retina e o resto vira COR_FUNDO. Com False (v2), o
# desfoque mistura a retina com o preto do preenchimento e cria faixas
# claras e escuras onde a câmera cortou o círculo (topo e base).
BEN_RESPEITA_MASCARA = True

# Erosão da máscara antes da normalização de Ben, em pixels da imagem
# final: descarta a borda da retina, onde a iluminação é irregular
EROSAO_MASCARA = 3

# Fração mínima da área esperada da retina coberta pela máscara.
# Abaixo disso a imagem é salva, mas marcada como "mascara_suspeita".
FRACAO_MINIMA_MASCARA = 0.8

# Fração do raio da retina mantida pela máscara circular. A borda do
# círculo costuma ter artefatos de iluminação, por isso é descartada.
RAIO_MASCARA = 0.9

# Cor do fundo fora da máscara. Com a normalização de Ben, o cinza 128
# é a cor "neutra"; sem ela, o fundo é preto.
COR_FUNDO = 128 if APLICAR_BEN else 0

# Raio mínimo, em pixels, para considerar que a retina foi encontrada
RAIO_MINIMO = 10

QUALIDADE_JPEG = 95

# v2: Otsu + Ben sem máscara. v3: máscara por limiar fixo + Ben só
# dentro da retina (sem as faixas no topo e na base).
VERSAO_PRE_PROCESSAMENTO = "v3"

# Pasta desta versão do pré-processamento
PASTA_IMAGENS_OTIMIZADAS = (
    f"/content/dataset_retina_{VERSAO_PRE_PROCESSAMENTO}_"
    f"{TAMANHO_IMAGEM[0]}"
)

# Zip da pasta acima no Drive. Se existir, a próxima sessão restaura
# as imagens em ~2 minutos em vez de extrair o dataset bruto (~30 min)
# e processá-lo de novo (~20 min).
CAMINHO_ZIP_CACHE = os.path.join(
    PASTA_CACHE_DRIVE,
    os.path.basename(PASTA_IMAGENS_OTIMIZADAS) + ".zip"
)

# Lista das imagens com máscara suspeita desta versão
CAMINHO_LISTA_SUSPEITAS = os.path.join(
    PASTA_CACHE_DRIVE,
    f"mascaras_suspeitas_{VERSAO_PRE_PROCESSAMENTO}.csv"
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

# Estado de uma imagem processada e salva, mas que merece inspeção
STATUS_SUSPEITO = "mascara_suspeita"

# ============================================================
# PIPELINE DE DADOS
# ============================================================

# Valor para EfficientNetB4 em 448 pixels numa GPU A100 com precisão
# mista. Numa L4, ou se houver erro de falta de memória (OOM), use 24
# ou 16.
TAMANHO_LOTE = 32

# Data augmentation aplicado somente no treino. Roda dentro do modelo,
# na GPU: no tf.data ele rodava na CPU e deixava a GPU ~85% do tempo
# esperando (1,2 s por lote no treino contra 0,1 s no predict).
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

# Opções: "EfficientNetB0", "EfficientNetB3", "EfficientNetB4",
# "EfficientNetB5", "EfficientNetV2S". Todas esperam a entrada na faixa de 0 a 255.
#
# Escolhido na primeira célula do notebook (variável de ambiente
# RETINOPATIA_BACKBONE), para que o mesmo commit treine mais de um
# backbone, cada um na sua pasta de resultados. Sem a variável, B4.
BACKBONE = os.environ.get("RETINOPATIA_BACKBONE", "EfficientNetB4")

# Precisão mista (float16 nos cálculos, float32 nos pesos): quase
# dobra a velocidade em GPUs L4/A100/T4 e reduz o uso de memória.
PRECISAO_MISTA = True

# Compilação XLA do passo de treino ("auto" = o Keras decide). Se o fit
# falhar ao compilar as camadas de data augmentation, use False.
COMPILACAO_XLA = "auto"

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

# ============================================================
# RESULTADOS DO EXPERIMENTO (no Drive)
# ============================================================


def _commit_do_codigo():
    """
    Commit do git em que este código está (o notebook faz checkout de
    uma revisão do GitHub). Fora de um repositório git, "sem_git".
    """

    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            check=True
        ).stdout.strip()

    except (OSError, subprocess.CalledProcessError):
        return "sem_git"


# Identifica a versão exata do código que gerou cada resultado: com ele,
# `git checkout <commit>` recupera o código de qualquer treino antigo.
COMMIT_CODIGO = _commit_do_codigo()

# Cada combinação de backbone, tamanho, versões e commit ganha a sua
# pasta, com pesos, histórico, predições, a configuração e uma cópia do
# código. Um treino com código novo nunca sobrescreve um antigo.
# Não faça git pull no meio de um experimento: o commit muda e, com o
# autoreload, a pasta dos pesos também.
NOME_EXPERIMENTO = (
    f"{BACKBONE}_{TAMANHO_IMAGEM[0]}"
    f"_pre{VERSAO_PRE_PROCESSAMENTO}"
    f"_split{VERSAO_SPLIT}"
    f"_seed{SEED}"
    f"_{COMMIT_CODIGO}"
)

PASTA_RESULTADOS = os.path.join(
    PASTA_PROJETO_DRIVE,
    "resultados",
    NOME_EXPERIMENTO
)

# Tabela com uma linha por treino concluído (commit, configuração
# principal e métricas de validação e teste), para comparar todos os
# experimentos num lugar só
CAMINHO_REGISTRO_EXPERIMENTOS = os.path.join(
    PASTA_PROJETO_DRIVE,
    "resultados",
    "registro_experimentos.csv"
)

# ============================================================
# TREINO — FASE 1 (apenas as cabeças da rede)
# ============================================================

# A Fase 1 só aquece as cabeças antes do fine-tuning. No treino v2, 7
# épocas (2,7 h) levaram a PR-AUC de validação a 0,59, e a primeira
# época da Fase 2 já a levou a 0,81.
EPOCAS_FASE1 = 2

TAXA_APRENDIZADO_FASE1 = 1e-3

CAMINHO_PESOS_FASE1 = os.path.join(
    PASTA_RESULTADOS,
    "melhor_modelo_fase1.weights.h5"
)

CAMINHO_HISTORICO_FASE1 = os.path.join(
    PASTA_RESULTADOS,
    "historico_fase1.csv"
)

PACIENCIA_EARLY_STOP_FASE1 = 2

# ============================================================
# TREINO — FASE 2 (fine-tuning)
# ============================================================

# No treino v2 a PR-AUC de validação estabilizou na época 9; as épocas
# seguintes não trouxeram ganho.
EPOCAS_FASE2 = 12

# Taxa máxima da Fase 2. Ela sobe linearmente durante o aquecimento e
# depois cai em cosseno até FRACAO_LR_FINAL dela. No v2, o
# ReduceLROnPlateau acompanhava val_loss e cortou a taxa em épocas em
# que a PR-AUC ainda melhorava.
TAXA_APRENDIZADO_FASE2 = 5e-5

EPOCAS_AQUECIMENTO_FASE2 = 1

FRACAO_LR_FINAL = 0.01

CAMINHO_PESOS_FASE2 = os.path.join(
    PASTA_RESULTADOS,
    "melhor_modelo_fase2.weights.h5"
)

CAMINHO_HISTORICO_FASE2 = os.path.join(
    PASTA_RESULTADOS,
    "historico_fase2.csv"
)

PACIENCIA_EARLY_STOP_FASE2 = 3

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

# Nome legível do modelo: rede, tamanho, versão do pré-processamento e
# commit do código, por exemplo "B4_448_v3_1ccd2cb"
NOME_MODELO = (
    f"{BACKBONE.replace('EfficientNet', '')}"
    f"_{TAMANHO_IMAGEM[0]}"
    f"_{VERSAO_PRE_PROCESSAMENTO}"
    f"_{COMMIT_CODIGO}"
)

CAMINHO_MODELO_FINAL = os.path.join(
    PASTA_MODELOS,
    NOME_MODELO + ".keras"
)

# ============================================================
# DIAGNÓSTICO
# ============================================================

# Reamostragens do bootstrap por paciente (intervalos de confiança)
NUMERO_BOOTSTRAP = 1000

# Grau mínimo da "retinopatia referenciável", reportada além do alvo
# binário do modelo (grau 0 contra graus 1 a 4)
GRAU_REFERENCIAVEL = 2

# Formas de combinar os dois olhos de um paciente comparadas no
# diagnóstico. "max" é a usada na avaliação.
AGREGACOES_PACIENTE = ["max", "media", "max_media"]

# Imagens mostradas na lista de falsos negativos
NUMERO_IMAGENS_ERRO = 12

# Modelo do treino v2 (EfficientNetB4 448, pré-processamento v2), para
# o diagnóstico sem retreinar (treino de ~10 h, código da tag treino-v2).
CAMINHO_MODELO_V2 = os.path.join(
    PASTA_MODELOS,
    "B4_448_v2_de5cc37.keras"
)

# Cache das imagens com o pré-processamento v2, recriado só para
# validação e teste, para avaliar o modelo v2 nas imagens com que ele
# foi treinado
PASTA_IMAGENS_V2 = f"/content/dataset_retina_v2_{TAMANHO_IMAGEM[0]}"

OPCOES_PRE_PROCESSAMENTO_V2 = {
    "metodo_mascara": "otsu",
    "ben_respeita_mascara": False
}

PASTA_DIAGNOSTICO_V2 = os.path.join(
    PASTA_PROJETO_DRIVE,
    "resultados",
    "diagnostico_modelo_v2"
)
