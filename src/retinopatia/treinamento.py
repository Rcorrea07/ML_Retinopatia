"""
Treino do modelo em duas fases.

Fase 1: a EfficientNet fica congelada e apenas as cabeças são treinadas,
com uma taxa de aprendizado alta, por poucas épocas (aquecimento).

Fase 2 (fine-tuning): a rede base é descongelada, mas as camadas de
BatchNormalization continuam congeladas — requisito conhecido ao fazer
fine-tuning de EfficientNet — e o treino segue com uma taxa de
aprendizado bem menor, que sobe durante um aquecimento e depois cai em
cosseno.

As duas fases usam callbacks com a mesma estrutura, criada por
`criar_callbacks`, mudando apenas os arquivos e a paciência. Pesos e
histórico vão para a pasta do experimento no Drive, para sobreviverem
a uma desconexão do Colab.

O desbalanceamento entre as classes é compensado com pesos por amostra
(`sample_weight`) acrescentados ao pipeline de treino: o argumento
`class_weight` do Keras não funciona em modelos com mais de uma saída.
"""

import os
import json

import numpy as np
import psutil
import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import (
    Callback,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint
)

from . import configuracao
from .modelo import compilar_modelo
from .persistencia import salvar_json


def calcular_pesos_classe(df_treino):
    """
    Calcula os pesos das classes de forma balanceada, para compensar o
    desbalanceamento entre imagens saudáveis e doentes.

    Retorna um dicionário no formato {0: peso, 1: peso}.
    """

    print("A calcular os pesos matemáticos das classes...")

    # Converte a coluna target para números inteiros
    coluna = "level_original" if configuracao.MODO_GRAU == "categorico" else "target"
    classes_reais_treino = (
        df_treino[coluna]
        .astype(int)
        .values
    )

    # Descobre quais classes realmente existem no treino
    # Neste projeto, o esperado é: [0, 1]
    classes = np.unique(classes_reais_treino)

    # Verifica se as duas classes estão presentes
    esperadas = set(range(5)) if coluna == "level_original" else {0, 1}
    if set(classes) != esperadas:
        raise ValueError(
            f"Classes inesperadas no treino: {classes}. "
            "O esperado era encontrar as classes 0 e 1."
        )

    # Calcula os pesos de acordo com a quantidade
    # de exemplos de cada classe
    pesos = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=classes_reais_treino
    )

    # Associa cada classe diretamente ao seu peso
    # Exemplo: {0: 0.65, 1: 2.10}
    if configuracao.PESOS_AMOSTRA == "raiz_balanceados_normalizada_media_1":
        pesos = np.sqrt(pesos)
        frequencias = np.array([(classes_reais_treino == c).mean() for c in classes])
        pesos = pesos / np.sum(pesos * frequencias)
    pesos_dicionario = dict(
        zip(
            classes.tolist(),
            pesos.tolist()
        )
    )

    print("\nPesos calculados:")

    for classe, peso in pesos_dicionario.items():
        print(
            f"Classe {classe} "
            f"({configuracao.NOMES_GRAUS[classe] if coluna == 'level_original' else configuracao.NOMES_CLASSES[classe]}): "
            f"peso {peso:.4f}"
        )

    print(
        "\nPesos calculados com sucesso! "
        "A classe menos frequente receberá maior peso durante o treinamento."
    )

    return pesos_dicionario


def adicionar_pesos_amostra(treino_data, pesos_dicionario):
    """
    Acrescenta a cada lote de treino o peso de cada amostra, de acordo
    com a sua classe binária, no formato (x, rótulos, pesos).

    O mesmo peso vale para as duas saídas do modelo.
    """

    pesos = tf.constant(
        [pesos_dicionario[i] for i in sorted(pesos_dicionario)],
        dtype=tf.float32
    )

    def com_pesos(x, rotulos):
        chave = (configuracao.SAIDA_CLASSES if configuracao.MODO_GRAU == "categorico"
                 else configuracao.SAIDA_DOENTE)
        classes = tf.cast(
            tf.reshape(rotulos[chave], [-1]),
            tf.int32
        )

        por_amostra = tf.gather(pesos, classes)
        return x, rotulos, {nome: por_amostra for nome in rotulos}

    return treino_data.map(
        com_pesos,
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)


class MonitorRecursos(Callback):
    """
    Acrescenta ao log de cada época a taxa de aprendizado em uso, o pico
    de memória da GPU e a RAM do sistema, em GB. Vai para o CSV do
    histórico e mostra quanta folga há para aumentar o lote, a
    resolução ou a rede.
    """

    def on_epoch_begin(self, epoca, logs=None):
        for gpu in tf.config.list_logical_devices("GPU"):
            tf.config.experimental.reset_memory_stats(gpu.name)

    def on_epoch_end(self, epoca, logs=None):
        if logs is None:
            return

        logs["taxa_aprendizado"] = float(
            self.model.optimizer.learning_rate
        )

        gpus = tf.config.list_logical_devices("GPU")

        for indice, gpu in enumerate(gpus):
            pico = tf.config.experimental.get_memory_info(gpu.name)["peak"]
            logs[f"memoria_gpu{indice}_pico_gb"] = pico / 1e9

        logs["memoria_ram_gb"] = psutil.virtual_memory().used / 1e9


def criar_callbacks(
    caminho_pesos,
    caminho_historico,
    paciencia_early_stop,
    metrica=configuracao.METRICA_MONITORADA
):
    """
    Cria os callbacks de monitorização do treino.

    O checkpoint e o early stopping acompanham a mesma métrica de
    validação (PR-AUC por padrão), e o histórico de cada época é
    gravado em CSV. A taxa de aprendizado não é ajustada por callback:
    na Fase 2 ela segue um agendamento em cosseno.
    """

    for caminho in (caminho_pesos, caminho_historico):
        os.makedirs(
            os.path.dirname(caminho),
            exist_ok=True
        )

    checkpoint = ModelCheckpoint(
        caminho_pesos,
        monitor=metrica,
        mode="max",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor=metrica,
        mode="max",
        patience=paciencia_early_stop,
        restore_best_weights=True,
        verbose=1
    )

    historico = CSVLogger(caminho_historico)

    # Antes do CSVLogger, para que os valores entrem no CSV
    return [
        MonitorRecursos(),
        checkpoint,
        early_stop,
        historico
    ]


def criar_callbacks_fase1():
    """
    Callbacks da Fase 1, com os valores definidos em `configuracao`.
    """

    return criar_callbacks(
        caminho_pesos=configuracao.CAMINHO_PESOS_FASE1,
        caminho_historico=configuracao.CAMINHO_HISTORICO_FASE1,
        paciencia_early_stop=configuracao.PACIENCIA_EARLY_STOP_FASE1
    )


def criar_callbacks_fase2():
    """
    Callbacks da Fase 2, com os valores definidos em `configuracao`.
    """

    return criar_callbacks(
        caminho_pesos=configuracao.CAMINHO_PESOS_FASE2,
        caminho_historico=configuracao.CAMINHO_HISTORICO_FASE2,
        paciencia_early_stop=configuracao.PACIENCIA_EARLY_STOP_FASE2
    )


def criar_taxa_cosseno(
    treino_data,
    epocas,
    taxa_maxima,
    epocas_aquecimento=configuracao.EPOCAS_AQUECIMENTO_FASE2,
    fracao_final=configuracao.FRACAO_LR_FINAL
):
    """
    Agendamento da taxa de aprendizado: sobe linearmente de quase zero
    até `taxa_maxima` nas épocas de aquecimento e depois cai em cosseno
    até `fracao_final * taxa_maxima` na última época.

    O aquecimento evita que os primeiros gradientes da rede recém-
    descongelada destruam os pesos da ImageNet.
    """

    passos_por_epoca = int(treino_data.cardinality())

    if passos_por_epoca <= 0:
        raise ValueError(
            "Não foi possível saber o número de lotes por época do "
            "pipeline de treino."
        )

    passos_aquecimento = passos_por_epoca * epocas_aquecimento

    return tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=taxa_maxima * fracao_final,
        decay_steps=passos_por_epoca * epocas - passos_aquecimento,
        alpha=fracao_final,
        warmup_target=taxa_maxima,
        warmup_steps=passos_aquecimento
    )


def treinar_fase1(
    modelo,
    treino_data,
    val_data,
    pesos_dicionario,
    callbacks,
    epocas=configuracao.EPOCAS_FASE1,
    caminho_pesos=configuracao.CAMINHO_PESOS_FASE1
):
    """
    Treina apenas as cabeças da rede e recarrega o melhor checkpoint ao
    final, deixando o modelo pronto para a avaliação.

    Retorna o histórico do treino.
    """

    print("Iniciando o treinamento...")

    historico = modelo.fit(
        # Os pesos por amostra compensam o desbalanceamento das classes
        adicionar_pesos_amostra(treino_data, pesos_dicionario),
        validation_data=val_data,
        epochs=epocas,
        callbacks=callbacks
    )

    print("\n Treino da Fase 1 concluído com sucesso!")

    print("Carregando o melhor checkpoint da Fase 1 para avaliação...")

    modelo.load_weights(caminho_pesos)

    return historico


def fine_tuning_fase2(
    modelo,
    modelo_base,
    treino_data,
    val_data,
    pesos_dicionario,
    callbacks,
    epocas=configuracao.EPOCAS_FASE2,
    taxa_aprendizado=configuracao.TAXA_APRENDIZADO_FASE2,
    caminho_pesos_fase1=configuracao.CAMINHO_PESOS_FASE1,
    caminho_pesos_fase2=configuracao.CAMINHO_PESOS_FASE2
):
    """
    Executa o fine-tuning: recupera os pesos da Fase 1, descongela a
    rede base mantendo a BatchNormalization congelada, recompila com uma
    taxa de aprendizado menor (cosseno com aquecimento) e treina
    novamente.

    Retorna o histórico do treino da Fase 2.
    """

    print("Recuperando os melhores pesos da Fase 1...")

    modelo.load_weights(caminho_pesos_fase1)

    print("Descongelando a rede e mantendo BatchNormalization congelada...")

    # Descongela o modelo base
    modelo_base.trainable = True

    # Mantém as camadas BatchNormalization congeladas
    for layer in modelo_base.layers:
        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):
            layer.trainable = False

    # Recompila obrigatoriamente depois de alterar trainable
    compilar_modelo(
        modelo,
        criar_taxa_cosseno(treino_data, epocas, taxa_aprendizado)
    )

    print("Iniciando o treinamento da Fase 2...")

    historico_fase2 = modelo.fit(
        adicionar_pesos_amostra(treino_data, pesos_dicionario),
        validation_data=val_data,
        epochs=epocas,
        callbacks=callbacks
    )

    print("Carregando o melhor checkpoint da Fase 2 para avaliação...")

    modelo.load_weights(caminho_pesos_fase2)

    return historico_fase2


def treinar_fase_monitorada(modelo, modelo_base, treino_data, val_data,
                            pesos_dicionario, fase):
    """Duas fases da campanha com snapshot completo e retomada explícita.

    Preserva os checkpoints/históricos legados. O snapshot adicional inclui o
    otimizador; o JSON preserva época, melhor métrica e espera do early stopping.
    Reiniciar não reproduz bit a bit a ordem aleatória anterior dos dados.
    """
    from pathlib import Path
    import pandas as pd
    from .modelo import GeM

    if fase not in (1, 2):
        raise ValueError("Fase deve ser 1 ou 2.")
    c = configuracao
    pasta = Path(c.PASTA_RESULTADOS)
    estado_caminho = pasta / f"estado_fase{fase}.json"
    ultimo = pasta / f"retomada_fase{fase}.keras"
    pesos = c.CAMINHO_PESOS_FASE1 if fase == 1 else c.CAMINHO_PESOS_FASE2
    historico_caminho = c.CAMINHO_HISTORICO_FASE1 if fase == 1 else c.CAMINHO_HISTORICO_FASE2
    epocas = c.EPOCAS_FASE1 if fase == 1 else c.EPOCAS_FASE2
    paciencia = c.PACIENCIA_EARLY_STOP_FASE1 if fase == 1 else c.PACIENCIA_EARLY_STOP_FASE2
    estado = json.loads(estado_caminho.read_text()) if estado_caminho.exists() else {}
    if estado and not c.RETOMAR:
        raise FileExistsError("Histórico de fase encontrado sem autorização de retomada.")
    if estado.get("concluida"):
        modelo.load_weights(pesos)
        return modelo, modelo_base, None
    epoca_inicial = int(estado.get("epocas_concluidas", 0))
    if epoca_inicial:
        modelo = tf.keras.models.load_model(ultimo, custom_objects={"GeM": GeM})
        modelo_base = modelo.get_layer(modelo_base.name)
        if os.path.isfile(historico_caminho):
            historico_salvo = pd.read_csv(historico_caminho)
            historico_salvo[historico_salvo["epoch"] < epoca_inicial].to_csv(
                historico_caminho, index=False)
    elif fase == 2:
        modelo.load_weights(c.CAMINHO_PESOS_FASE1)
        modelo_base.trainable = True
        for camada in modelo_base.layers:
            if isinstance(camada, tf.keras.layers.BatchNormalization):
                camada.trainable = False
        compilar_modelo(modelo, criar_taxa_cosseno(
            treino_data, epocas, c.TAXA_APRENDIZADO_FASE2))

    callbacks = criar_callbacks(pesos, historico_caminho, paciencia)
    checkpoint, early_stop = callbacks[1:3]
    # O carregamento explícito abaixo restaura o melhor checkpoint inclusive
    # quando ele foi obtido antes da desconexão.
    early_stop.restore_best_weights = False
    callbacks[-1] = CSVLogger(historico_caminho, append=bool(epoca_inicial))

    class EstadoRetomada(Callback):
        def on_train_begin(self, logs=None):
            if "melhor_metrica" in estado:
                checkpoint.best = estado["melhor_metrica"]
                early_stop.best = estado["melhor_metrica"]
                early_stop.wait = estado.get("espera", 0)

        def on_epoch_end(self, epoch, logs=None):
            if not np.isfinite(logs.get(c.METRICA_MONITORADA, np.nan)):
                raise RuntimeError("Métrica monitorada ausente ou não finita.")
            temporario = str(ultimo).replace(".keras", ".parcial.keras")
            self.model.save(temporario)
            os.replace(temporario, ultimo)
            salvar_json(estado_caminho, {
                "epocas_concluidas": epoch + 1,
                "melhor_metrica": float(checkpoint.best),
                "espera": int(early_stop.wait), "concluida": False})

    callbacks.append(EstadoRetomada())
    historico = modelo.fit(
        adicionar_pesos_amostra(treino_data, pesos_dicionario),
        validation_data=val_data, epochs=epocas, initial_epoch=epoca_inicial,
        callbacks=callbacks)
    estado_final = json.loads(estado_caminho.read_text())
    estado_final["concluida"] = True
    salvar_json(estado_caminho, estado_final)
    modelo.load_weights(pesos)
    return modelo, modelo_base, historico
