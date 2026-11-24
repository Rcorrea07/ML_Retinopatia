"""
F10: o RETFound-Green comparado com o R12, em PyTorch.

O RETFound-Green (Engelmann e Bernabeu, arXiv 2405.00117) é uma ViT-S/14
pré-treinada pelo DinoV2 e depois em 75 mil fotos de retina. Aqui ela é
ajustada para a mesma tarefa do R12 (retinopatia sim ou não, com a
cabeça auxiliar de grau), com os mesmos dados de treino (EyePACS, extras
de 2015 e datasets externos), e medida na mesma validação, pela mesma
avaliação (`avaliacao.avaliar_predicoes`). A comparação com o R12 é
pareada nos mesmos pacientes (`analise.comparar_pareado`).

Licença dos pesos: só pesquisa não comercial, sem projeto com empresa
envolvida. É só pesquisa da IC: nada deste modelo, nem um derivado dele
(destilação), vai para o app dos postos sem permissão escrita do autor.

Decisões (23/11/2026):
- as imagens são as dos caches de 768 px (com a normalização de Ben),
  reduzidas para a entrada de 392 da rede. O pré-treino viu fotos sem
  Ben, então ela pode render menos do que renderia com fotos cruas;
- a receita segue a do Keras onde faz sentido (duas fases, cosseno com
  aquecimento, pesos das classes, perda de Huber no grau, o mesmo
  aumento de dados, checkpoint pela PR-AUC da validação) e a de
  fine-tuning de ViT onde a rede pede (AdamW, taxa por camada,
  stochastic depth, corte do gradiente, bfloat16);
- o teste do EyePACS fica fechado: ele já foi gasto no R09.

Este módulo não importa TensorFlow, e o PyTorch só é importado dentro
das funções: o pacote continua importável sem ele.
"""

import math
import os
import time
import urllib.request

import numpy as np
import pandas as pd

from . import ambiente_colab, configuracao

ROTULO = "RETFound-Green"


# ============================================================
# DADOS
# ============================================================


def preparar_conjuntos():
    """
    Treino e validação do experimento, com as imagens no disco: as do
    EyePACS (cache do Drive ou extração), as fotos só de treino (extras
    de 2015 e datasets externos) e a redução da fumaça. É a preparação
    de `execucao.preparar_dados` sem os ajustes que usam TensorFlow
    (pares e destilação, que este experimento não usa).

    Retorna:
        (df_treino, df_val)
    """

    from . import conjunto_dados, execucao, fontes

    if configuracao.MODELO_PARES or configuracao.DESTILACAO:
        raise ValueError("O RETFound-Green não tem modelo de pares nem destilação.")

    dados, df_treino, df_val, df_teste, _ = execucao.preparar_imagens_eyepacs()

    df_adicionais = fontes.preparar_dados_adicionais(dados)

    if df_adicionais is not None:
        df_treino = conjunto_dados.ampliar_treino(df_treino, df_adicionais)

    df_treino, df_val, _ = conjunto_dados.reduzir_para_fumaca(
        df_treino,
        df_val,
        df_teste
    )

    return df_treino, df_val


def caminhos_imagens(df):
    """
    Caminho de cada imagem processada: a pasta da linha (coluna
    `pasta_processada`, no treino ampliado) ou a pasta do EyePACS.
    """

    pastas = (
        df["pasta_processada"]
        if "pasta_processada" in df
        else [configuracao.PASTA_IMAGENS_OTIMIZADAS] * len(df)
    )

    return [os.path.join(pasta, nome) for pasta, nome in zip(pastas, df["image"])]


def pesos_classes(classes):
    """
    Pesos balanceados das classes 0 e 1 (os de
    `compute_class_weight("balanced")`): n / (2 · contagem da classe).
    """

    classes = np.asarray(classes).astype(int)

    contagens = np.bincount(classes, minlength=2)

    if (contagens == 0).any():
        raise ValueError(f"O treino precisa das duas classes: contagens {contagens}.")

    return {classe: len(classes) / (2 * contagens[classe]) for classe in (0, 1)}


def aumentar(img, gerador, parametros=None, cor_fundo=None):
    """
    O aumento de dados do treino do Keras (`pipeline_dados.criar_aumento_dados`)
    numa imagem RGB: espelhos horizontal e vertical, rotação de até
    ±180°, zoom de ±15% (o fundo exposto fica cinza 128), brilho e
    contraste. Devolve float32 na faixa 0–255.
    """

    import cv2

    if parametros is None:
        parametros = configuracao.AUMENTO_DADOS

    if cor_fundo is None:
        cor_fundo = configuracao.COR_FUNDO

    if gerador.random() < 0.5:
        img = img[:, ::-1]

    if gerador.random() < 0.5:
        img = img[::-1]

    altura, largura = img.shape[:2]

    matriz = cv2.getRotationMatrix2D(
        (largura / 2, altura / 2),
        gerador.uniform(-1, 1) * parametros["fator_rotacao"] * 360,
        1 + gerador.uniform(-1, 1) * parametros["fator_zoom"]
    )

    img = cv2.warpAffine(
        np.ascontiguousarray(img),
        matriz,
        (largura, altura),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(cor_fundo, cor_fundo, cor_fundo)
    ).astype(np.float32)

    img += gerador.uniform(-1, 1) * parametros["fator_brilho"] * 255

    media = img.mean(axis=(0, 1), keepdims=True)
    img = (img - media) * (1 + gerador.uniform(-1, 1) * parametros["fator_contraste"]) + media

    return np.clip(img, 0, 255)


class ImagensRetina:
    """
    Conjunto de imagens para o DataLoader do PyTorch, na ordem do
    DataFrame: lê o cache (falha se o tamanho não for o do experimento),
    reduz para a entrada da rede e, no treino, aplica o aumento de dados.
    Cada item: (imagem float32 C×A×L em 0–255, classe, grau, peso).
    """

    def __init__(self, df, treino, pesos=None, seed=None):
        self.caminhos = caminhos_imagens(df)
        self.classes = df["target"].astype(np.float32).values
        self.graus = df["level_original"].astype(np.float32).values
        self.treino = treino

        pesos = pesos or {0: 1.0, 1: 1.0}
        self.pesos = np.array([pesos[int(c)] for c in self.classes], dtype=np.float32)

        # Cada processo do DataLoader troca este gerador pelo seu
        # (`iniciar_trabalhador`)
        self.gerador = np.random.default_rng(
            configuracao.SEED_TREINO if seed is None else seed
        )

    def __len__(self):
        return len(self.caminhos)

    def __getitem__(self, indice):
        import cv2

        caminho = self.caminhos[indice]

        img = cv2.imread(caminho, cv2.IMREAD_COLOR)

        if img is None:
            raise FileNotFoundError(f"Imagem ilegível ou ausente: {caminho}")

        if img.shape[:2] != tuple(configuracao.TAMANHO_IMAGEM):
            raise ValueError(
                f"{caminho} tem {img.shape[:2]}, mas o cache do experimento é "
                f"{configuracao.TAMANHO_IMAGEM}."
            )

        img = cv2.resize(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            tuple(configuracao.TAMANHO_ENTRADA_RETFOUND)[::-1],
            interpolation=cv2.INTER_AREA
        )

        img = aumentar(img, self.gerador) if self.treino else img.astype(np.float32)

        return (
            np.ascontiguousarray(img.transpose(2, 0, 1)),
            self.classes[indice],
            self.graus[indice],
            self.pesos[indice]
        )


def iniciar_trabalhador(_):
    """
    Dá a cada processo do DataLoader um gerador próprio, derivado da
    semente do PyTorch (os processos herdariam o mesmo).
    """

    import torch

    informacao = torch.utils.data.get_worker_info()

    informacao.dataset.gerador = np.random.default_rng(informacao.seed % 2**32)


def criar_carregador(df, treino, pesos=None):
    """
    DataLoader de um conjunto: embaralha só no treino (a validação fica
    na ordem do DataFrame, que liga as predições às linhas).
    """

    import torch

    trabalhadores = configuracao.TRABALHADORES_DADOS_RETFOUND

    return torch.utils.data.DataLoader(
        ImagensRetina(df, treino, pesos),
        batch_size=configuracao.TAMANHO_LOTE_RETFOUND,
        shuffle=treino,
        drop_last=treino,
        num_workers=trabalhadores,
        worker_init_fn=iniciar_trabalhador,
        generator=torch.Generator().manual_seed(configuracao.SEED_TREINO),
        pin_memory=True,
        persistent_workers=trabalhadores > 0,
        prefetch_factor=4 if trabalhadores > 0 else None
    )


# ============================================================
# REDE
# ============================================================


def baixar_pesos(caminho=None, url=None):
    """
    Os pesos do RETFound-Green (87 MB, release v0.1 do repositório
    oficial), baixados uma vez por sessão.
    """

    if caminho is None:
        caminho = configuracao.CAMINHO_PESOS_RETFOUND

    if url is None:
        url = configuracao.URL_PESOS_RETFOUND

    if not os.path.isfile(caminho):
        os.makedirs(os.path.dirname(caminho), exist_ok=True)

        print(f"Baixando os pesos do RETFound-Green de {url}...")

        urllib.request.urlretrieve(url, caminho + ".parcial")
        os.replace(caminho + ".parcial", caminho)

    print(f"Pesos do RETFound-Green: {caminho} ({os.path.getsize(caminho) / 1e6:.1f} MB)")

    return caminho


def construir_rede(caminho_pesos=None):
    """
    O RETFound-Green (timm, média dos tokens de imagem, embedding de 384)
    com as duas cabeças do projeto: `doente` (logit) e `grau`
    (regressão). A rede recebe a imagem em 0–255 e normaliza por dentro
    (média 0,5 e desvio 0,5 na escala 0–1). Com `caminho_pesos` None,
    pesos aleatórios (testes).
    """

    import timm
    import torch
    from torch import nn

    extrator = timm.create_model(
        configuracao.ARQUITETURA_RETFOUND,
        img_size=tuple(configuracao.TAMANHO_ENTRADA_RETFOUND),
        num_classes=0,
        drop_path_rate=configuracao.TAXA_DROP_PATH_RETFOUND,
        checkpoint_path=caminho_pesos or ""
    )

    extrator.reset_classifier(0, global_pool="avg")

    class RedeRetfound(nn.Module):

        def __init__(self):
            super().__init__()

            media, desvio = configuracao.MEDIA_DESVIO_RETFOUND

            self.extrator = extrator
            self.dropout = nn.Dropout(configuracao.TAXA_DROPOUT)
            self.doente = nn.Linear(extrator.num_features, 1)
            self.grau = nn.Linear(extrator.num_features, 1)

            self.register_buffer("media", torch.tensor(media))
            self.register_buffer("desvio", torch.tensor(desvio))

        def forward(self, imagens):
            x = (imagens / 255.0 - self.media) / self.desvio

            caracteristicas = self.dropout(self.extrator(x))

            return (
                self.doente(caracteristicas).squeeze(1),
                self.grau(caracteristicas).squeeze(1)
            )

    return RedeRetfound()


def camada_do_parametro(nome, numero_blocos):
    """
    Profundidade de um parâmetro para a taxa por camada: 0 para a
    entrada (patch, posição e tokens), i + 1 para o bloco i, e
    `numero_blocos` + 1 para a norma final e as cabeças.
    """

    if not nome.startswith("extrator."):
        return numero_blocos + 1

    nome = nome[len("extrator."):]

    if nome.startswith(("cls_token", "reg_token", "pos_embed", "patch_embed")):
        return 0

    if nome.startswith("blocks."):
        return int(nome.split(".")[1]) + 1

    return numero_blocos + 1


def grupos_parametros(
    parametros_nomeados,
    numero_blocos,
    taxa,
    decaimento_camadas,
    decaimento_pesos
):
    """
    Grupos do AdamW: a taxa de cada camada é `taxa` ·
    `decaimento_camadas`^(distância até o topo), e vieses, normalizações,
    tokens e posições não têm decaimento de pesos. Cada grupo guarda a
    taxa máxima em `taxa_maxima`, que o cosseno multiplica.
    """

    grupos = {}

    for nome, parametro in parametros_nomeados:
        if not parametro.requires_grad:
            continue

        camada = camada_do_parametro(nome, numero_blocos)

        sem_decaimento = parametro.ndim == 1 or nome.endswith(("token", "pos_embed"))

        grupo = grupos.setdefault((camada, sem_decaimento), {
            "params": [],
            "taxa_maxima": taxa * decaimento_camadas ** (numero_blocos + 1 - camada),
            "weight_decay": 0.0 if sem_decaimento else decaimento_pesos
        })

        grupo["params"].append(parametro)

    for grupo in grupos.values():
        grupo["lr"] = grupo["taxa_maxima"]

    return list(grupos.values())


def fator_cosseno(passo, total, aquecimento, fracao_final):
    """
    Fração da taxa máxima no `passo`: sobe linearmente durante o
    `aquecimento` e depois cai em cosseno até `fracao_final` no último
    passo (a mesma forma de `treinamento.criar_taxa_cosseno`).
    """

    if passo < aquecimento:
        return (passo + 1) / aquecimento

    progresso = min(1.0, (passo - aquecimento) / max(1, total - aquecimento))

    return fracao_final + (1 - fracao_final) * 0.5 * (1 + math.cos(math.pi * progresso))


# ============================================================
# TREINO
# ============================================================


def _ambiente_torch():
    """
    GPU e versões do PyTorch e do timm, para o configuracao.json.
    """

    import timm
    import torch

    return {
        "gpu": [
            torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())
        ],
        "versao_torch": torch.__version__,
        "versao_timm": timm.__version__
    }


def configurar_sementes(seed=None):
    """
    Sementes do Python, NumPy e PyTorch. O cuDNN escolhe os algoritmos
    mais rápidos (não determinísticos), como no treino do Keras sem
    determinismo completo.
    """

    import random

    import torch

    if seed is None:
        seed = configuracao.SEED_TREINO

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.benchmark = True


def _perda(logits, graus_previstos, classes, graus, pesos):
    """
    A perda do Keras: BCE da cabeça `doente` mais `PESO_PERDA_GRAU` ·
    Huber da cabeça de grau, as duas ponderadas pelo peso da classe de
    cada amostra e com média no lote.
    """

    from torch.nn import functional

    binaria = functional.binary_cross_entropy_with_logits(
        logits,
        classes,
        reduction="none"
    )

    grau = functional.huber_loss(
        graus_previstos,
        graus,
        reduction="none",
        delta=configuracao.DELTA_HUBER
    )

    return ((binaria + configuracao.PESO_PERDA_GRAU * grau) * pesos).mean()


def _transformacoes_tta():
    """
    As variantes da TTA na ordem do Keras (`avaliacao._variantes_tta`):
    original, espelho horizontal, espelho vertical e os dois.
    """

    import torch

    return [
        lambda x: x,
        lambda x: torch.flip(x, dims=[3]),
        lambda x: torch.flip(x, dims=[2]),
        lambda x: torch.flip(x, dims=[2, 3])
    ]


def prever_variantes(rede, carregador, numero_variantes=4):
    """
    Probabilidades e graus de cada variante da TTA, na ordem do
    carregador, montados nos modos de `avaliacao.MODOS_TTA` (só os que
    cabem em `numero_variantes`).

    Retorna:
        {modo: (probabilidades, graus_previstos)}
    """

    import torch

    from . import avaliacao

    dispositivo = next(rede.parameters()).device
    transformacoes = _transformacoes_tta()[:numero_variantes]

    probabilidades = [[] for _ in transformacoes]
    graus = [[] for _ in transformacoes]

    rede.eval()

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for imagens, *_ in carregador:
            imagens = imagens.to(dispositivo, non_blocking=True)

            for indice, transformar in enumerate(transformacoes):
                logits, graus_previstos = rede(transformar(imagens))

                probabilidades[indice].append(torch.sigmoid(logits.float()).cpu().numpy())
                graus[indice].append(graus_previstos.float().cpu().numpy())

    variantes = [
        (np.concatenate(p), np.concatenate(g))
        for p, g in zip(probabilidades, graus)
    ]

    return {
        modo: (
            np.mean([v[0] for v in variantes[:quantidade]], axis=0),
            np.mean([v[1] for v in variantes[:quantidade]], axis=0)
        )
        for modo, quantidade in avaliacao.MODOS_TTA.items()
        if quantidade <= numero_variantes
    }


def _medir_epoca(rede, carregador_val, df_val):
    """
    PR-AUC e ROC-AUC por olho na validação, sem TTA: o monitor do
    checkpoint, como `val_doente_pr_auc` no Keras.
    """

    from sklearn.metrics import average_precision_score, roc_auc_score

    probabilidades, _ = prever_variantes(rede, carregador_val, numero_variantes=1)["sem_tta"]

    classes = df_val["target"].astype(int).values

    return {
        "val_pr_auc": average_precision_score(classes, probabilidades),
        "val_roc_auc": roc_auc_score(classes, probabilidades)
    }


def _salvar_historico(linhas, fase):
    """
    O histórico da fase até a última época, regravado a cada época
    (sobrevive a uma desconexão).
    """

    caminho = os.path.join(configuracao.PASTA_RESULTADOS, f"historico_fase{fase}.csv")

    pd.DataFrame(linhas).to_csv(caminho, index=False)


def treinar_fase(
    rede,
    fase,
    carregador_treino,
    carregador_val,
    df_val,
    otimizador,
    epocas,
    paciencia,
    aquecimento_epocas=0
):
    """
    Uma fase de treino: taxa em cosseno por passo (com aquecimento), uma
    linha por época (perda, PR-AUC e ROC-AUC da validação, taxa, tempo,
    memória) e os pesos da melhor PR-AUC da validação restaurados no fim,
    com parada antecipada depois de `paciencia` épocas sem melhora.

    Retorna o histórico (DataFrame).
    """

    import copy

    import psutil
    import torch

    dispositivo = next(rede.parameters()).device
    passos_por_epoca = len(carregador_treino)
    total = epocas * passos_por_epoca
    aquecimento = aquecimento_epocas * passos_por_epoca

    parametros = [p for grupo in otimizador.param_groups for p in grupo["params"]]

    melhor, melhor_estado, sem_melhora = -np.inf, None, 0
    linhas = []
    passo = 0

    for epoca in range(1, epocas + 1):
        inicio = time.time()
        torch.cuda.reset_peak_memory_stats()

        rede.train()

        # Na Fase 1 o extrator está congelado e fica em modo de avaliação
        # (sem stochastic depth), como o backbone com training=False no Keras
        if fase == 1:
            rede.extrator.eval()

        # Somada na GPU: um .item() por passo pararia a GPU a cada lote
        soma_perda, lotes = torch.zeros((), device=dispositivo), 0

        for imagens, classes, graus, pesos in carregador_treino:
            fator = (
                fator_cosseno(passo, total, aquecimento, configuracao.FRACAO_LR_FINAL)
                if fase == 2 else 1.0
            )

            for grupo in otimizador.param_groups:
                grupo["lr"] = grupo["taxa_maxima"] * fator

            imagens = imagens.to(dispositivo, non_blocking=True)
            classes, graus, pesos = (
                t.to(dispositivo, non_blocking=True).float() for t in (classes, graus, pesos)
            )

            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, graus_previstos = rede(imagens)

            perda = _perda(logits.float(), graus_previstos.float(), classes, graus, pesos)

            if not torch.isfinite(perda):
                raise FloatingPointError(f"Perda não finita na fase {fase}, época {epoca}.")

            otimizador.zero_grad(set_to_none=True)
            perda.backward()
            torch.nn.utils.clip_grad_norm_(parametros, configuracao.LIMITE_GRADIENTE_RETFOUND)
            otimizador.step()

            soma_perda += perda.detach()
            lotes += 1
            passo += 1

        medidas = _medir_epoca(rede, carregador_val, df_val)

        linha = {
            "fase": fase,
            "epoca": epoca,
            "perda": soma_perda.item() / max(1, lotes),
            **medidas,
            "taxa_maxima_atual": max(g["lr"] for g in otimizador.param_groups),
            "segundos": round(time.time() - inicio, 1),
            "memoria_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "ram_gb": round(psutil.virtual_memory().used / 1e9, 1)
        }

        linhas.append(linha)
        _salvar_historico(linhas, fase)

        print(
            f"Fase {fase}, época {epoca}/{epocas}: perda {linha['perda']:.4f}, "
            f"val PR-AUC {linha['val_pr_auc']:.4f}, val ROC-AUC {linha['val_roc_auc']:.4f}, "
            f"{linha['segundos']:.0f} s, GPU {linha['memoria_gpu_gb']} GB"
        )

        if medidas["val_pr_auc"] > melhor:
            melhor, sem_melhora = medidas["val_pr_auc"], 0
            melhor_estado = copy.deepcopy(
                {chave: valor.detach().cpu() for chave, valor in rede.state_dict().items()}
            )
        else:
            sem_melhora += 1

            if sem_melhora >= paciencia:
                print(f"Parada antecipada: {paciencia} épocas sem melhora da PR-AUC.")
                break

    rede.load_state_dict(melhor_estado)

    print(f"Fase {fase}: pesos da melhor época restaurados (val PR-AUC {melhor:.4f}).")

    return pd.DataFrame(linhas)


def treinar(rede, carregador_treino, carregador_val, df_val):
    """
    Fase 1 (só as cabeças, extrator congelado, taxa constante) e Fase 2
    (a rede inteira, taxa por camada em cosseno com aquecimento).

    Retorna:
        (histórico da Fase 1, histórico da Fase 2)
    """

    import torch

    for parametro in rede.extrator.parameters():
        parametro.requires_grad = False

    cabecas = [p for nome, p in rede.named_parameters() if not nome.startswith("extrator.")]

    otimizador = torch.optim.AdamW(
        [{
            "params": cabecas,
            "lr": configuracao.TAXA_APRENDIZADO_FASE1,
            "taxa_maxima": configuracao.TAXA_APRENDIZADO_FASE1
        }],
        weight_decay=configuracao.DECAIMENTO_PESOS_RETFOUND
    )

    historico_fase1 = treinar_fase(
        rede,
        1,
        carregador_treino,
        carregador_val,
        df_val,
        otimizador,
        configuracao.EPOCAS_FASE1,
        configuracao.PACIENCIA_EARLY_STOP_FASE1
    )

    for parametro in rede.extrator.parameters():
        parametro.requires_grad = True

    otimizador = torch.optim.AdamW(
        grupos_parametros(
            rede.named_parameters(),
            len(rede.extrator.blocks),
            configuracao.TAXA_APRENDIZADO_RETFOUND,
            configuracao.DECAIMENTO_CAMADAS_RETFOUND,
            configuracao.DECAIMENTO_PESOS_RETFOUND
        ),
        betas=(0.9, 0.999)
    )

    historico_fase2 = treinar_fase(
        rede,
        2,
        carregador_treino,
        carregador_val,
        df_val,
        otimizador,
        configuracao.EPOCAS_FASE2,
        configuracao.PACIENCIA_EARLY_STOP_FASE2,
        aquecimento_epocas=configuracao.EPOCAS_AQUECIMENTO_FASE2
    )

    return historico_fase1, historico_fase2


def conferir_entrada(carregador):
    """
    Confere um lote antes do treino: formato da entrada da rede e escala
    0–255 (sem normalização por engano), como `verificar_escala` no
    Keras.
    """

    imagens, classes, *_ = next(iter(carregador))

    esperado = (3,) + tuple(configuracao.TAMANHO_ENTRADA_RETFOUND)

    if tuple(imagens.shape[1:]) != esperado:
        raise ValueError(f"Lote com formato {tuple(imagens.shape[1:])}, esperado {esperado}.")

    if imagens.max() <= 1.0:
        raise ValueError("As imagens chegaram em 0–1; a rede espera 0–255.")

    print(
        f"Lote conferido: {tuple(imagens.shape)}, faixa "
        f"{imagens.min():.0f}–{imagens.max():.0f}, {int(classes.sum())} doentes"
    )


# ============================================================
# COMPARAÇÃO COM O R12
# ============================================================


def comparar_com_r12(caminho_predicoes, caminho_r12=None):
    """
    O RETFound-Green contra o modelo final (R12 semente 42) nos mesmos
    pacientes da validação, nos três modos de TTA
    (`analise.comparar_pareado`). A regra fixada antes de rodar: o
    RETFound-Green ganha se a especificidade por paciente com TTA2 for
    maior e o intervalo pareado da diferença ficar acima de zero.

    Retorna:
        DataFrame indexado por (modo, métrica)
    """

    from . import analise

    if caminho_r12 is None:
        caminho_r12 = configuracao.CAMINHO_PREDICOES_VALIDACAO_R09

    partes = {
        modo: analise.comparar_pareado(
            analise.predicoes_do_modo(caminho_predicoes, modo),
            analise.predicoes_do_modo(caminho_r12, modo)
        )
        for modo in analise.MODOS
    }

    comparacao = pd.concat(partes, names=["modo", "metrica"])

    print("\n" + "=" * 60)
    print(f"{ROTULO} (A) contra o R12 semente 42 (B), validação por paciente")
    print("=" * 60)
    print(comparacao.round(4).to_string())

    principal = comparacao.loc[(configuracao.MODO_TTA, "especificidade")]

    if principal["ic95_inferior"] > 0:
        veredito = f"o {ROTULO} ganha"
    elif principal["ic95_superior"] < 0:
        veredito = "o R12 ganha"
    else:
        veredito = "empate (o intervalo da diferença inclui zero)"

    print(
        f"\nRegra ({configuracao.MODO_TTA}, especificidade por paciente): "
        f"diferença {principal['diferenca']:+.4f} "
        f"[{principal['ic95_inferior']:+.4f}; {principal['ic95_superior']:+.4f}] → {veredito}."
    )
    print(
        "O R12 semente 42 foi escolhido entre duas sementes por esta mesma "
        "especificidade: a régua favorece um pouco o R12."
    )

    return comparacao


# ============================================================
# A EXECUÇÃO INTEIRA
# ============================================================


def executar():
    """
    O experimento inteiro numa sessão: dados, pesos, rede, as duas
    fases, o modelo salvo, a avaliação na validação (a de sempre, com os
    três modos de TTA), o resumo da execução, a sensibilidade por grau e
    a comparação pareada com o R12. O teste não é aberto.

    Retorna um dicionário com as partes.
    """

    import torch

    from . import avaliacao, diagnostico, utilitarios

    if not os.path.isdir(configuracao.PASTA_PROJETO_DRIVE):
        raise RuntimeError("O Drive não está montado: monte-o antes de rodar.")

    if not torch.cuda.is_available():
        raise RuntimeError("Sem GPU: escolha um ambiente com GPU no Colab.")

    if configuracao.AVALIAR_TESTE:
        raise RuntimeError("O teste do EyePACS já foi gasto no R09: rode com AVALIAR_TESTE desligado.")

    print("Experimento:", configuracao.NOME_EXPERIMENTO)

    configurar_sementes()

    caminho_pesos = baixar_pesos()

    df_treino, df_val = preparar_conjuntos()

    print(f"Treino: {len(df_treino)} imagens; validação: {len(df_val)}.")

    pesos = pesos_classes(df_treino["target"])

    print(f"Pesos das classes: {pesos}")

    carregador_treino = criar_carregador(df_treino, True, pesos)
    carregador_val = criar_carregador(df_val, False)

    conferir_entrada(carregador_treino)

    rede = construir_rede(caminho_pesos).cuda()

    utilitarios.salvar_configuracao(ambiente_ml=_ambiente_torch())

    historico_fase1, historico_fase2 = treinar(rede, carregador_treino, carregador_val, df_val)

    os.makedirs(os.path.dirname(configuracao.CAMINHO_MODELO_RETFOUND), exist_ok=True)
    torch.save(rede.state_dict(), configuracao.CAMINHO_MODELO_RETFOUND)

    print(f"Modelo salvo em: {configuracao.CAMINHO_MODELO_RETFOUND}")

    print("\nGerando probabilidades na validação (quatro variantes da TTA)...")

    resultado = avaliacao.avaliar_predicoes(
        df_val,
        prever_variantes(rede, carregador_val),
        "validacao_fase_2",
        f"{ROTULO} — validação",
        "validação",
        modo_tta=configuracao.MODO_TTA
    )

    avaliacao.registrar_experimento(
        resultado,
        extras={
            "tamanho_lote": configuracao.TAMANHO_LOTE_RETFOUND,
            "tamanho_entrada": configuracao.TAMANHO_ENTRADA_RETFOUND[0],
            "modelo_avaliado": os.path.basename(configuracao.CAMINHO_MODELO_RETFOUND)
        }
    )

    por_grau = diagnostico.relatorio_por_grau(
        resultado["tabela"],
        resultado["limiar_olho"],
        "validação"
    )

    comparacao = None

    if configuracao.FUMACA:
        print("\nFumaça: a comparação com o R12 precisa da validação inteira e fica de fora.")

    else:
        comparacao = comparar_com_r12(
            os.path.join(configuracao.PASTA_RESULTADOS, "predicoes_validacao_fase_2.csv")
        )

        comparacao.to_csv(os.path.join(configuracao.PASTA_RESULTADOS, "comparacao_r12.csv"))

    return {
        "validacao": resultado,
        "historico_fase1": historico_fase1,
        "historico_fase2": historico_fase2,
        "por_grau": por_grau,
        "comparacao_r12": comparacao
    }


def rodar(desligar_ao_terminar=True):
    """
    `executar` e, com `desligar_ao_terminar`, a sessão encerrada no fim,
    também se algo der errado.
    """

    try:
        return executar()

    finally:
        if desligar_ao_terminar:
            ambiente_colab.desligar_ambiente()


if __name__ == "__main__":
    rodar(desligar_ao_terminar=False)
