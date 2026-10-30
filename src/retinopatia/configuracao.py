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
from datetime import datetime

from . import experimentos

# ============================================================
# EXPERIMENTO
# ============================================================

# Escolhido na primeira célula do notebook (variável de ambiente
# RETINOPATIA_EXPERIMENTO). As constantes que o experimento muda, listadas
# em `experimentos.py`, substituem os valores padrão abaixo antes de os
# caminhos derivados serem calculados.
EXPERIMENTO = experimentos.nome_experimento_ativo()

_SOBREPOSICOES = experimentos.sobreposicoes(EXPERIMENTO)
_SOBREPOSICOES_USADAS = set()

# Execução de fumaça (RETINOPATIA_FUMACA=1): poucos pacientes e uma
# época por fase, só para conferir no Colab que um recurso novo roda,
# salva e recarrega antes de gastar horas de GPU. As métricas dela não
# valem nada, e a pasta de resultados leva o sufixo "_fumaca".
FUMACA = os.environ.get("RETINOPATIA_FUMACA", "0") == "1"


def _valor(nome, padrao):
    """
    Valor da constante `nome`: o do experimento ativo, se ele a mudar,
    ou o padrão. No fim do módulo, uma constante do experimento que não
    passou por aqui (nome errado) para a execução.
    """

    if nome in _SOBREPOSICOES:
        _SOBREPOSICOES_USADAS.add(nome)
        return _SOBREPOSICOES[nome]

    return padrao


# ============================================================
# REPRODUTIBILIDADE
# ============================================================

# A divisão treino/validação/teste usa sempre a mesma semente: todos os
# treinos são comparados nos mesmos pacientes, e os CSVs de split salvos
# continuam válidos
SEED_SPLIT = 42

# Semente dos pesos, do embaralhamento e do augmentation. Repetir um
# experimento com outra semente (RETINOPATIA_SEED_TREINO) mede a variação
# do treino sem mudar os pacientes de cada conjunto.
SEED_TREINO = int(os.environ.get(
    "RETINOPATIA_SEED_TREINO",
    _valor("SEED_TREINO", 42)
))

SEED = SEED_TREINO

# Operações determinísticas no TensorFlow. Garantem o mesmo resultado a
# cada execução, mas podem deixar o treino mais lento ou falhar em
# operações sem implementação determinística na GPU. Se
# `pipeline_dados.medir_velocidade` ou o tempo por época mostrarem
# gargalo, ou se aparecer erro de determinismo, desligue.
OPERACOES_DETERMINISTICAS = True

# ============================================================
# CAMINHOS (ambiente do Google Colab)
# ============================================================

# O dataset vem de um compartilhamento da conta pessoal do responsável.
# Na conta do projeto ele é um atalho neste caminho; outra conta pode
# apontar o seu caminho em RETINOPATIA_ZIP_DATASET.
CAMINHO_ZIP_PRINCIPAL = os.environ.get(
    "RETINOPATIA_ZIP_DATASET",
    "/content/drive/MyDrive/Datasets/diabetic-retinopathy-detection.zip"
)

# Datasets de retinopatia além do EyePACS (docs/DATASETS.md), guardados
# compactados na pasta Datasets da conta pessoal (5 TB), compartilhada
# para leitura com a conta do projeto. O notebook Baixar_Datasets, rodado
# na conta pessoal, baixa direto para ela, sem passar pelo computador.
PASTA_DATASETS = os.path.dirname(CAMINHO_ZIP_PRINCIPAL)

# Do Kaggle: pasta -> identificador do dataset. O DDR é a cópia já
# recortada e sem as fotos não graduáveis (licença do original:
# CC BY-NC-SA 4.0, sem uso comercial); o IDRiD é CC BY 4.0. Do Messidor-2
# vêm só os rótulos de consenso do Google: as fotos pedem cadastro na
# ADCIS e ficam para a validação externa.
DATASETS_KAGGLE = {
    "DDR": "mariaherrerot/ddrdataset",
    "IDRiD": "aaryapatel98/indian-diabetic-retinopathy-image-dataset",
    "Messidor2_rotulos": "google-brain/messidor2-dr-grades",
}

# De endereços públicos: pasta -> (arquivo, endereço). O DeepDRiD (CC BY-SA
# 4.0) fica no próprio GitHub, ~1,3 GB.
DATASETS_URL = {
    "DeepDRiD": (
        "DeepDRiD.zip",
        "https://github.com/deepdrdoc/DeepDRiD/archive/refs/heads/master.zip"
    ),
}

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

# Zips das imagens já processadas, um por versão do pré-processamento.
# Aqui ficam os caches de 448 (v3) criados até outubro de 2026.
PASTA_CACHE_DRIVE = os.path.join(
    PASTA_DADOS_INTERNOS,
    "cache"
)

# Caches novos (512 px ou mais, ~7 a 11 GB cada) não cabem no Drive do
# projeto. Ficam no Drive pessoal do responsável, numa pasta compartilhada
# como leitura com a conta do projeto, onde aparece como atalho com o
# mesmo nome em "Meu Drive". O mesmo caminho vale nas duas contas.
# Ler pelo atalho não gasta cota do projeto; gravar gastaria (o arquivo
# conta na cota de quem grava), por isso só o notebook Preparar_Caches,
# com o Drive pessoal montado, grava ali (RETINOPATIA_GRAVAR_CACHE=1).
PASTA_CACHE_COMPARTILHADA = "/content/drive/MyDrive/ProjetoRetinopatia_caches"

GRAVAR_CACHE = os.environ.get("RETINOPATIA_GRAVAR_CACHE", "0") == "1"

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

# As soluções vencedoras usaram de 384 a 768 pixels: lesões pequenas,
# como microaneurismas (grau 1), somem em 224 pixels.
TAMANHO_IMAGEM = tuple(_valor("TAMANHO_IMAGEM", (448, 448)))

# Medidas em pixels do pré-processamento foram calibradas em 448 e
# crescem com a imagem, para que a receita seja a mesma em qualquer
# resolução (em 448 os valores são exatamente os da v3)
_ESCALA_IMAGEM = TAMANHO_IMAGEM[0] / 448

# Normalização de cor de Ben Graham (vencedor de 2015):
# 4 * imagem - 4 * desfoque_gaussiano(imagem) + 128.
# Remove as diferenças de iluminação entre câmeras e realça lesões.
APLICAR_BEN = True

SIGMA_BEN = _valor("SIGMA_BEN", 10 * _ESCALA_IMAGEM)

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
EROSAO_MASCARA = round(3 * _ESCALA_IMAGEM)

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
# e processá-lo de novo (~20 min). É procurado primeiro na pasta
# compartilhada e depois na do projeto (caches de 448).
_NOME_ZIP_CACHE = os.path.basename(PASTA_IMAGENS_OTIMIZADAS) + ".zip"

CAMINHOS_ZIP_CACHE = [
    os.path.join(PASTA_CACHE_COMPARTILHADA, _NOME_ZIP_CACHE),
    os.path.join(PASTA_CACHE_DRIVE, _NOME_ZIP_CACHE)
]

# Onde um cache novo é gravado (só com GRAVAR_CACHE)
CAMINHO_ZIP_CACHE = CAMINHOS_ZIP_CACHE[0]

# Pasta da lista de máscaras suspeitas: junto do cache quando ele é
# gerado na conta pessoal; nos dados internos do projeto nos demais casos
_PASTA_LISTAS = (
    PASTA_CACHE_COMPARTILHADA if GRAVAR_CACHE else PASTA_CACHE_DRIVE
)

# Lista das imagens com máscara suspeita desta versão (o nome sem o
# tamanho é o dos caches de 448, mantido como estava)
_SUFIXO_TAMANHO = "" if TAMANHO_IMAGEM[0] == 448 else f"_{TAMANHO_IMAGEM[0]}"

CAMINHO_LISTA_SUSPEITAS = os.path.join(
    _PASTA_LISTAS,
    f"mascaras_suspeitas_{VERSAO_PRE_PROCESSAMENTO}{_SUFIXO_TAMANHO}.csv"
)

# Limita o número de threads para evitar sobrecarga (a A100 do Colab
# tem 12 núcleos)
MAXIMO_WORKERS = 12

# ============================================================
# DADOS EXTRAS (test da competição de 2015, só para o treino)
# ============================================================

# O zip do Kaggle tem 88.702 imagens: 35.126 do "train" (as únicas
# usadas até a versão 3) e 53.576 do "test". Os rótulos do test não
# vêm no zip: foram publicados depois pelos organizadores, no fórum da
# competição, e são os mesmos que o TensorFlow Datasets usa. Com True,
# as imagens do test entram só no treino; a validação e o teste do
# projeto continuam os mesmos, para comparar com os treinos anteriores.
USAR_DADOS_EXTRA = _valor("USAR_DADOS_EXTRA", True)

URL_ROTULOS_EXTRA = (
    "https://storage.googleapis.com/kaggle-forum-message-attachments/"
    "90528/2877/retinopathy_solution.csv"
)

# Cópia dos rótulos no Drive (baixada uma vez)
CAMINHO_ROTULOS_EXTRA = (
    "/content/drive/MyDrive/Datasets/retinopathy_solution.csv"
)

PASTA_IMAGENS_EXTRA = os.path.join(
    PASTA_BASE,
    "test"
)

PASTA_IMAGENS_OTIMIZADAS_EXTRA = PASTA_IMAGENS_OTIMIZADAS + "_extra"

_NOME_ZIP_CACHE_EXTRA = (
    os.path.basename(PASTA_IMAGENS_OTIMIZADAS_EXTRA) + ".zip"
)

CAMINHOS_ZIP_CACHE_EXTRA = [
    os.path.join(PASTA_CACHE_COMPARTILHADA, _NOME_ZIP_CACHE_EXTRA),
    os.path.join(PASTA_CACHE_DRIVE, _NOME_ZIP_CACHE_EXTRA)
]

CAMINHO_ZIP_CACHE_EXTRA = CAMINHOS_ZIP_CACHE_EXTRA[0]

CAMINHO_LISTA_SUSPEITAS_EXTRA = os.path.join(
    _PASTA_LISTAS,
    f"mascaras_suspeitas_{VERSAO_PRE_PROCESSAMENTO}{_SUFIXO_TAMANHO}_extra.csv"
)

# ============================================================
# DATASETS EXTERNOS (DDR, APTOS, DeepDRiD, IDRiD; só para o treino)
# ============================================================

# Datasets de outros países e câmeras (docs/DATASETS.md), na pasta
# Datasets do Drive, cada um no zip em que foi baixado (notebook
# Baixar_Datasets). Como as fotos extras de 2015, entram só no treino:
# a validação e o teste continuam os do EyePACS. O experimento escolhe
# quais usar, pelos nomes abaixo.
ZIPS_DATASETS_EXTERNOS = {
    "DDR": os.path.join(PASTA_DATASETS, "DDR", "ddrdataset.zip"),
    "APTOS": os.path.join(
        PASTA_DATASETS,
        "aptos2019-blindness-detection.zip"
    ),
    "DeepDRiD": os.path.join(PASTA_DATASETS, "DeepDRiD", "DeepDRiD.zip"),
    "IDRiD": os.path.join(
        PASTA_DATASETS,
        "IDRiD",
        "indian-diabetic-retinopathy-image-dataset.zip"
    ),
}

DATASETS_EXTERNOS = list(_valor("DATASETS_EXTERNOS", []))

_EXTERNOS_DESCONHECIDOS = set(DATASETS_EXTERNOS) - set(ZIPS_DATASETS_EXTERNOS)

if _EXTERNOS_DESCONHECIDOS:
    raise ValueError(
        f"Datasets externos desconhecidos: {sorted(_EXTERNOS_DESCONHECIDOS)}. "
        f"Opções: {sorted(ZIPS_DATASETS_EXTERNOS)}"
    )

# Fotos originais tiradas dos zips, apagadas depois de processadas
PASTA_EXTERNOS_BRUTOS = "/content/datasets_externos"

# Pasta das imagens processadas, zips do cache e lista de máscaras
# suspeitas de cada dataset, nos mesmos lugares dos do EyePACS
PASTAS_IMAGENS_EXTERNAS = {
    nome: f"{PASTA_IMAGENS_OTIMIZADAS}_{nome.lower()}"
    for nome in ZIPS_DATASETS_EXTERNOS
}

CAMINHOS_ZIP_CACHE_EXTERNOS = {
    nome: [
        os.path.join(pasta_cache, os.path.basename(pasta) + ".zip")
        for pasta_cache in (PASTA_CACHE_COMPARTILHADA, PASTA_CACHE_DRIVE)
    ]
    for nome, pasta in PASTAS_IMAGENS_EXTERNAS.items()
}

CAMINHOS_LISTA_SUSPEITAS_EXTERNOS = {
    nome: os.path.join(
        _PASTA_LISTAS,
        f"mascaras_suspeitas_{VERSAO_PRE_PROCESSAMENTO}{_SUFIXO_TAMANHO}"
        f"_{nome.lower()}.csv"
    )
    for nome in ZIPS_DATASETS_EXTERNOS
}

# No DeepDRiD o grau é dado por olho, não por foto: as duas fotos do
# mesmo olho levam o mesmo grau, e numa foto recusada pela nota de
# qualidade ("Overall quality" 0, mais da metade delas) as lesões podem
# não aparecer. Entram só as fotos com nota de qualidade aceita (1).
QUALIDADE_MINIMA_DEEPDRID = 1

# Fração das fotos de um dataset externo que pode falhar no
# pré-processamento (fotos ilegíveis ou sem retina) e ficar fora do
# treino; acima dela a execução para, como no EyePACS
FRACAO_MAXIMA_FALHAS_EXTERNOS = 0.01

# Sufixo dos nomes de experimento e de modelo quando os dados extras ou
# os datasets externos são usados (os resultados não se misturam com os
# treinos só com o train). A lista dos externos fica no configuracao.json.
SUFIXO_DADOS = (
    ("_extra" if USAR_DADOS_EXTRA else "")
    + ("_externos" if DATASETS_EXTERNOS else "")
)

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
# ou 16. A B0 em 448 usou só 4,5 GB da GPU com 32.
TAMANHO_LOTE = _valor("TAMANHO_LOTE", 32)

# Data augmentation aplicado somente no treino. Roda dentro do modelo,
# na GPU: no tf.data ele rodava na CPU e deixava a GPU ~85% do tempo
# esperando (1,2 s por lote no treino contra 0,1 s no predict).
# A retina não tem orientação canônica, então rotações de 360 graus
# e espelhamentos são seguros.
AUMENTO_DADOS = _valor("AUMENTO_DADOS", {
    "fator_rotacao": 0.5,       # Fração de volta completa: 0.5 = ±180 graus
    "fator_zoom": 0.15,         # Zoom aleatório de ±15%
    "fator_brilho": 0.1,        # Brilho de ±10% da faixa 0 a 255
    "fator_contraste": 0.1      # Contraste entre 0.9 e 1.1
})

# ============================================================
# MODELO
# ============================================================

FORMATO_ENTRADA = TAMANHO_IMAGEM + (3,)

# Opções: "EfficientNetB0", "EfficientNetB3", "EfficientNetB4",
# "EfficientNetB5", "EfficientNetV2B0", "EfficientNetV2S". Todas esperam
# a entrada na faixa de 0 a 255.
#
# Definido pelo experimento; sem ele, pela variável de ambiente
# RETINOPATIA_BACKBONE (forma antiga de escolher). Sem nenhum dos dois, B4.
BACKBONE = _valor(
    "BACKBONE",
    os.environ.get("RETINOPATIA_BACKBONE", "EfficientNetB4")
)

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
PESO_PERDA_GRAU = _valor("PESO_PERDA_GRAU", 0.5)

# Perda de Huber (SmoothL1) na regressão do grau: menos sensível a
# rótulos errados do que o erro quadrático.
DELTA_HUBER = 1.0

# Modelo de pares de olhos (De Fauw, o_O): a mesma rede, com pesos
# compartilhados, vê os dois olhos do paciente; cada olho é previsto a
# partir das características dele e das do outro olho. As saídas
# continuam por olho, com os mesmos nomes. No celular são duas
# passagens da mesma rede mais uma cabeça pequena.
MODELO_PARES = _valor("MODELO_PARES", False)

# Destilação: a B4 448 com dados extras (melhor modelo único) gera,
# antes do treino e sem TTA, a probabilidade de cada imagem do treino.
# A perda binária do aluno passa a ser
# PESO_ROTULO_REAL * BCE(rótulo) + (1 - PESO_ROTULO_REAL) * BCE(professor).
# O professor não faz parte do modelo entregue.
DESTILACAO = _valor("DESTILACAO", False)

PESO_ROTULO_REAL = _valor("PESO_ROTULO_REAL", 0.7)

SAIDA_PROFESSOR = "professor"

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

# Data e hora da execução, fixadas na primeira importação e guardadas no
# ambiente do processo: o autoreload reimporta este módulo, mas a pasta
# do treino em andamento não muda
ID_EXECUCAO = os.environ.setdefault(
    "RETINOPATIA_ID_EXECUCAO",
    datetime.now().strftime("%Y%m%d_%H%M")
)

# Cada execução ganha a sua pasta, com pesos, histórico, predições, a
# configuração e uma cópia do código: nada é sobrescrito, nem repetindo
# o mesmo experimento no mesmo commit.
# Não faça git pull no meio de um experimento: o código mudaria no meio
# do treino.
NOME_EXPERIMENTO = (
    f"{EXPERIMENTO}"
    f"_{BACKBONE}_{TAMANHO_IMAGEM[0]}"
    f"_pre{VERSAO_PRE_PROCESSAMENTO}"
    f"_seed{SEED_TREINO}"
    f"{SUFIXO_DADOS}"
    f"_{COMMIT_CODIGO}"
    f"_{ID_EXECUCAO}"
    f"{'_fumaca' if FUMACA else ''}"
)

PASTA_RESULTADOS_GERAL = os.path.join(
    PASTA_PROJETO_DRIVE,
    "resultados"
)

PASTA_RESULTADOS = os.path.join(
    PASTA_RESULTADOS_GERAL,
    NOME_EXPERIMENTO
)

# Os pesos da melhor época de cada fase só servem dentro da sessão (o
# modelo final completo vai para modelos/). Ficam no disco local: no
# Drive, eram 100 a 400 MB repetidos por treino.
PASTA_CHECKPOINTS = os.path.join(
    "/content/checkpoints",
    NOME_EXPERIMENTO
)

# Cada execução grava só o seu resumo (resumo.csv na sua pasta). A tabela
# geral, com uma linha por treino, é atualizada por
# `avaliacao.consolidar_experimentos()`, chamada uma vez depois das
# sessões paralelas: dois treinos gravando o mesmo CSV ao mesmo tempo
# perderiam linhas.
CAMINHO_REGISTRO_EXPERIMENTOS = os.path.join(
    PASTA_RESULTADOS_GERAL,
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
    PASTA_CHECKPOINTS,
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
EPOCAS_FASE2 = _valor("EPOCAS_FASE2", 12)

# Taxa máxima da Fase 2. Ela sobe linearmente durante o aquecimento e
# depois cai em cosseno até FRACAO_LR_FINAL dela. No v2, o
# ReduceLROnPlateau acompanhava val_loss e cortou a taxa em épocas em
# que a PR-AUC ainda melhorava.
TAXA_APRENDIZADO_FASE2 = _valor("TAXA_APRENDIZADO_FASE2", 5e-5)

EPOCAS_AQUECIMENTO_FASE2 = 1

# Redimensionamento progressivo (o_O treinou 128 → 256 → 512): caminho
# de um modelo .keras já treinado em resolução menor. Os pesos dele
# iniciam a rede (a arquitetura não depende do tamanho da imagem), a
# Fase 1 é pulada e a Fase 2 parte desses pesos. None = ImageNet.
PESOS_INICIAIS = _valor("PESOS_INICIAIS", None)

if FUMACA:
    EPOCAS_FASE1 = EPOCAS_FASE2 = 1
    EPOCAS_AQUECIMENTO_FASE2 = 0

FRACAO_LR_FINAL = 0.01

CAMINHO_PESOS_FASE2 = os.path.join(
    PASTA_CHECKPOINTS,
    "melhor_modelo_fase2.weights.h5"
)

CAMINHO_HISTORICO_FASE2 = os.path.join(
    PASTA_RESULTADOS,
    "historico_fase2.csv"
)

PACIENCIA_EARLY_STOP_FASE2 = _valor("PACIENCIA_EARLY_STOP_FASE2", 3)

# ============================================================
# AVALIAÇÃO
# ============================================================

# Contexto de rastreio médico: falsos negativos custam mais caro que
# falsos positivos, por isso o limiar é escolhido pela sensibilidade.
SENSIBILIDADE_DESEJADA = 0.90

# Test-time augmentation usada na avaliação principal (limiares e
# relatórios): "sem_tta", "tta2" (original e espelho horizontal) ou
# "tta4" (mais os espelhos vertical e nos dois eixos). Os três modos são
# previstos de uma vez e comparados em todas as avaliações.
MODO_TTA = _valor("MODO_TTA", "tta4")

if MODO_TTA not in ("sem_tta", "tta2", "tta4"):
    raise ValueError(f"MODO_TTA desconhecido: {MODO_TTA} (sem_tta, tta2 ou tta4).")

# O teste só mede o modelo final escolhido. Nos experimentos ele fica
# fechado: tudo é comparado e escolhido na validação, e abrir o teste a
# cada treino acabaria orientando as escolhas por ele. Para o modelo
# final, RETINOPATIA_AVALIAR_TESTE=1 na primeira célula do notebook.
AVALIAR_TESTE = os.environ.get("RETINOPATIA_AVALIAR_TESTE", "0") == "1"


# ============================================================
# MODELO FINAL
# ============================================================

# Nome legível do modelo: experimento, rede, tamanho, versão do
# pré-processamento, commit e execução, por exemplo
# "R01_b0_512_B0_512_v3_extra_ab12cd3_20261022_1430". Os modelos antigos
# (antes dos experimentos nomeados) não têm o experimento nem a execução.
# Uma semente de treino diferente da padrão entra no nome: dois treinos do
# mesmo experimento na mesma sessão (execucao.treinar_em_paralelo)
# gravariam o mesmo arquivo.
SUFIXO_SEMENTE = "" if SEED_TREINO == SEED_SPLIT else f"_seed{SEED_TREINO}"

NOME_MODELO = (
    f"{EXPERIMENTO}_"
    f"{BACKBONE.replace('EfficientNet', '')}"
    f"_{TAMANHO_IMAGEM[0]}"
    f"_{VERSAO_PRE_PROCESSAMENTO}"
    f"{SUFIXO_DADOS}"
    f"{SUFIXO_SEMENTE}"
    f"_{COMMIT_CODIGO}"
    f"_{ID_EXECUCAO}"
)

# O modelo de uma execução de fumaça fica na pasta dela, fora de modelos/
CAMINHO_MODELO_FINAL = os.path.join(
    PASTA_RESULTADOS if FUMACA else PASTA_MODELOS,
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

# ============================================================
# PROFESSOR DA DESTILAÇÃO
# ============================================================

# B4 448, pré-processamento v3, dados extras (commit 22f95c4): melhor
# modelo único até outubro de 2026 (teste, por paciente: ROC-AUC 0,923)
CAMINHO_PROFESSOR = os.path.join(
    PASTA_MODELOS,
    "B4_448_v3_extra_22f95c4.keras"
)

# O professor vê as imagens no tamanho em que foi treinado, do cache de
# 448 (v3), mesmo quando o aluno usa outra resolução
TAMANHO_IMAGEM_PROFESSOR = (448, 448)

PASTA_IMAGENS_PROFESSOR = "/content/dataset_retina_v3_448"

# ============================================================
# MODELO FINAL ESCOLHIDO (R09, modelo_final.py)
# ============================================================

# B0 768 (R10): o melhor por olho nas duas sementes e empatado por
# paciente com os outros candidatos, e mais leve no celular que a V2-B0.
# Das duas sementes, a de maior especificidade por paciente na validação
# com TTA2 (0,790 contra 0,785).
CAMINHO_MODELO_ESCOLHIDO = _valor(
    "CAMINHO_MODELO_ESCOLHIDO",
    os.path.join(
        PASTA_MODELOS,
        "R10_b0_768_B0_768_v3_extra_c35c12a_20261027_1920.keras"
    )
)

# Versões para o celular (TFLite) do modelo escolhido
PASTA_TFLITE = os.path.join(
    PASTA_MODELOS,
    "tflite"
)

# Imagens da validação usadas para conferir que o TFLite prevê o mesmo
# que o Keras (as primeiras do conjunto, sem TTA)
AMOSTRAS_PARIDADE_TFLITE = 200

# Maior diferença de probabilidade aceita entre o modelo de inferência
# (float32, sem augmentation) e o modelo treinado (precisão mista)
TOLERANCIA_INFERENCIA = 0.02

# ============================================================
# VÁRIOS TREINOS NA MESMA GPU (execucao.treinar_em_paralelo)
# ============================================================

# Memória reservada por treino para decidir quantos rodam ao mesmo tempo.
# Medido no R10 (B0 768, lote 32): pico de 12,6 GB de VRAM e 25 GB de
# RAM. Uma A100 do Colab (80 GB de VRAM, 167 GB de RAM) comporta três.
VRAM_POR_TREINO_GB = 16
RAM_POR_TREINO_GB = 27
FOLGA_RAM_GB = 10

# De quanto em quanto tempo o notebook mostra o andamento dos treinos
INTERVALO_ACOMPANHAMENTO_MIN = 10

# Saída de cada treino (uma linha por época), copiada para o Drive
PASTA_LOGS_LOCAL = "/content/logs_treinos"

PASTA_LOGS_PARALELOS = os.path.join(
    PASTA_RESULTADOS_GERAL,
    "logs_paralelos"
)

# ============================================================
# CONFERÊNCIA DO EXPERIMENTO
# ============================================================

# Uma constante do experimento que nenhuma linha acima leu tem o nome
# errado (ou ainda não pode ser mudada por experimento): parar aqui evita
# um treino inteiro com a configuração padrão achando que é outra.
_NAO_APLICADAS = set(_SOBREPOSICOES) - _SOBREPOSICOES_USADAS

if _NAO_APLICADAS:
    raise ValueError(
        f"O experimento {EXPERIMENTO} muda constantes que não existem ou "
        f"não podem ser mudadas por experimento: {sorted(_NAO_APLICADAS)}"
    )

if MODELO_PARES and DESTILACAO:
    raise ValueError(
        "Destilação e modelo de pares ainda não foram implementados juntos."
    )

# O professor lê as imagens dos caches de 448 do EyePACS
if DATASETS_EXTERNOS and DESTILACAO:
    raise ValueError(
        "Destilação e datasets externos ainda não foram implementados juntos."
    )

# Constantes mudadas por este experimento, gravadas no configuracao.json
SOBREPOSICOES_EXPERIMENTO = dict(_SOBREPOSICOES)
