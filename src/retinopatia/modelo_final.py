"""
R09: o modelo escolhido medido no teste, uma única vez, e convertido
para o celular (TFLite).

Não treina nada. Carrega o .keras escolhido (`CAMINHO_MODELO_ESCOLHIDO`),
processa só as imagens da validação e do teste (~7 mil, poucos minutos),
escolhe os limiares na validação com o modo de TTA do experimento e os
aplica ao teste. Os três modos de TTA também são medidos no teste, cada
um com o seu limiar da validação, para decidir quantas passagens pela
rede o celular faz.

Depois, monta a versão de inferência (float32, sem augmentation, saída
[probabilidade, grau]) e a converte para TFLite em float32, float16 e
int8 dinâmico (pesos em int8), conferindo em imagens da validação que
cada arquivo prevê o mesmo que o modelo Keras.

O teste só abre com RETINOPATIA_AVALIAR_TESTE=1, na primeira célula do
notebook Modelo_Final. Rode uma vez só.
"""

import os
import shutil
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from . import (
    ambiente_colab,
    avaliacao,
    configuracao,
    conjunto_dados,
    diagnostico,
    pipeline_dados,
    pre_processamento,
    utilitarios,
)
from .modelo import carregar_modelo_salvo, construir_modelo_inferencia

# Formatos do TFLite gerados: float32 (referência), float16 (metade do
# tamanho) e int8 dinâmico (pesos em int8, ativações em float)
FORMATOS_TFLITE = ["fp32", "fp16", "int8"]


def _conversor_tflite(pasta_savedmodel, formato):
    """
    Conversor TFLite do SavedModel, ajustado para o formato.
    """

    conversor = tf.lite.TFLiteConverter.from_saved_model(pasta_savedmodel)

    if formato in ("fp16", "int8"):
        conversor.optimizations = [tf.lite.Optimize.DEFAULT]

    if formato == "fp16":
        conversor.target_spec.supported_types = [tf.float16]

    return conversor


def preparar_imagens_avaliacao():
    """
    Extrai as fotos do train e processa só as da validação e do teste,
    no tamanho do experimento. A lista de máscaras suspeitas vai para a
    pasta desta execução, sem sobrescrever a do treino.

    Retorna:
        (df_val, df_teste)
    """

    os.makedirs(configuracao.PASTA_RESULTADOS, exist_ok=True)

    cache_restaurado = ambiente_colab.restaurar_cache_imagens()

    ambiente_colab.extrair_dataset(somente_rotulos=cache_restaurado)

    _, _, df_val, df_teste = conjunto_dados.preparar_dataset()

    pre_processamento.pre_processar_dataset(
        pd.concat([df_val, df_teste]),
        caminho_lista_suspeitas=os.path.join(
            configuracao.PASTA_RESULTADOS,
            "mascaras_suspeitas_validacao_teste.csv"
        )
    )

    return df_val, df_teste


def avaliar_modelo_final(
    df_val,
    df_teste,
    caminho_modelo=configuracao.CAMINHO_MODELO_ESCOLHIDO,
    modo_tta=configuracao.MODO_TTA
):
    """
    Limiares na validação e medida única no teste do modelo salvo em
    `caminho_modelo`, com o diagnóstico completo (erros por grau,
    referenciável, intervalos de confiança) no teste.

    Retorna:
        (modelo, resultado_validacao, resultado_teste)
    """

    print(f"Modelo avaliado: {caminho_modelo}")

    modelo = carregar_modelo_salvo(caminho_modelo)

    if tuple(modelo.input_shape[1:]) != tuple(configuracao.FORMATO_ENTRADA):
        raise ValueError(
            f"O modelo espera {modelo.input_shape[1:]}, mas o experimento "
            f"prepara imagens {configuracao.FORMATO_ENTRADA}."
        )

    val_data = pipeline_dados.criar_dataset(df_val, treino=False)
    teste_data = pipeline_dados.criar_dataset(df_teste, treino=False)

    utilitarios.salvar_configuracao()

    resultado_validacao = avaliacao.avaliar_modelo(
        modelo,
        val_data,
        df_val,
        None,
        "Modelo final",
        modo_tta=modo_tta
    )

    resultado_teste = avaliacao.avaliar_no_teste(
        modelo,
        teste_data,
        df_teste,
        resultado_validacao,
        liberado=configuracao.AVALIAR_TESTE
    )

    avaliacao.registrar_experimento(
        resultado_validacao,
        resultado_teste,
        extras={"modelo_avaliado": os.path.basename(caminho_modelo)}
    )

    diagnostico.relatorio_completo(resultado_validacao, resultado_teste)

    return modelo, resultado_validacao, resultado_teste


def _amostra_validacao(df_val, quantidade):
    """
    As primeiras `quantidade` imagens da validação (já processadas),
    como um array em 0–255, e as classes delas.
    """

    amostra = df_val.head(quantidade)

    imagens = np.concatenate([
        lote.numpy()
        for lote, _ in pipeline_dados.criar_dataset(amostra, treino=False)
    ])

    return imagens, amostra["target"].astype(int).values


def _prever_keras(modelo, imagens):
    """
    Probabilidades de retinopatia do modelo treinado, sem TTA.
    """

    saida = modelo.predict(imagens, verbose=0)

    # Um modelo carregado do disco pode devolver lista em vez de dicionário
    if not isinstance(saida, dict):
        saida = dict(zip(modelo.output_names, saida))

    return np.asarray(saida[configuracao.SAIDA_DOENTE]).ravel()


def _prever_tflite(conteudo, imagens):
    """
    Probabilidades do arquivo TFLite, imagem por imagem, e o tempo médio
    por imagem (CPU do Colab: só uma referência, não o celular).
    """

    interpretador = tf.lite.Interpreter(model_content=conteudo, num_threads=os.cpu_count())
    interpretador.allocate_tensors()

    entrada = interpretador.get_input_details()[0]["index"]
    saida = interpretador.get_output_details()[0]["index"]

    probabilidades = []
    inicio = time.perf_counter()

    for imagem in imagens:
        interpretador.set_tensor(entrada, imagem[np.newaxis].astype(np.float32))
        interpretador.invoke()
        probabilidades.append(interpretador.get_tensor(saida)[0, 0])

    segundos = (time.perf_counter() - inicio) / len(imagens)

    return np.asarray(probabilidades), segundos


def exportar_tflite(
    modelo,
    df_val,
    pasta=configuracao.PASTA_TFLITE,
    caminho_modelo=configuracao.CAMINHO_MODELO_ESCOLHIDO,
    amostras=configuracao.AMOSTRAS_PARIDADE_TFLITE,
    tolerancia=configuracao.TOLERANCIA_INFERENCIA
):
    """
    Converte o modelo para TFLite (float32, float16 e int8 dinâmico) e
    confere, em `amostras` imagens da validação sem TTA, que cada versão
    prevê o mesmo que o modelo Keras: diferença de probabilidade, ROC-AUC
    na amostra, tamanho do arquivo e tempo por imagem na CPU do Colab.

    Entrada de cada arquivo: float32 [1, altura, largura, 3], a imagem
    processada em 0–255. Saída: float32 [1, 2], probabilidade de
    retinopatia e grau previsto (0 a 4).

    Retorna a tabela de paridade (também salva na pasta da execução).
    """

    inferencia = construir_modelo_inferencia(modelo)

    imagens, classes = _amostra_validacao(df_val, amostras)

    referencia = _prever_keras(modelo, imagens)

    diferenca = np.abs(inferencia.predict(imagens, verbose=0)[:, 0] - referencia).max()

    print(f"Modelo de inferência (float32, sem augmentation): diferença máxima {diferenca:.5f}")

    if diferenca > tolerancia:
        raise RuntimeError(
            f"O modelo de inferência difere do treinado em {diferenca:.4f} "
            f"(tolerância {tolerancia}): a cópia dos pesos falhou."
        )

    pasta_savedmodel = "/content/modelo_inferencia_savedmodel"
    shutil.rmtree(pasta_savedmodel, ignore_errors=True)
    inferencia.export(pasta_savedmodel)

    os.makedirs(pasta, exist_ok=True)

    nome_base = os.path.splitext(os.path.basename(caminho_modelo))[0]

    linhas = [{
        "formato": "keras",
        "arquivo": os.path.basename(caminho_modelo),
        "tamanho_mb": round(os.path.getsize(caminho_modelo) / 1e6, 1),
        "dif_media": 0.0,
        "dif_maxima": 0.0,
        "roc_auc_amostra": roc_auc_score(classes, referencia),
        "segundos_por_imagem_cpu": None
    }]

    for formato in FORMATOS_TFLITE:
        conteudo = _conversor_tflite(pasta_savedmodel, formato).convert()

        caminho = os.path.join(pasta, f"{nome_base}_{formato}.tflite")

        with open(caminho, "wb") as arquivo:
            arquivo.write(conteudo)

        probabilidades, segundos = _prever_tflite(conteudo, imagens)
        diferencas = np.abs(probabilidades - referencia)

        linhas.append({
            "formato": formato,
            "arquivo": os.path.basename(caminho),
            "tamanho_mb": round(len(conteudo) / 1e6, 1),
            "dif_media": diferencas.mean(),
            "dif_maxima": diferencas.max(),
            "roc_auc_amostra": roc_auc_score(classes, probabilidades),
            "segundos_por_imagem_cpu": segundos
        })

        print(f"{formato}: {caminho} ({linhas[-1]['tamanho_mb']} MB)")

    tabela = pd.DataFrame(linhas).set_index("formato")

    print(f"\nParidade com o Keras em {len(imagens)} imagens da validação (sem TTA):")
    print(tabela.round(4).to_string())

    tabela.to_csv(os.path.join(configuracao.PASTA_RESULTADOS, "paridade_tflite.csv"))

    return tabela


def rodar_final(desligar_ao_terminar=True):
    """
    O R09 inteiro numa sessão: imagens da validação e do teste, medida
    única no teste e TFLite. Com `desligar_ao_terminar`, encerra a sessão
    no fim, também se algo der errado.

    Retorna:
        (resultado_validacao, resultado_teste, tabela_tflite)
    """

    if not configuracao.AVALIAR_TESTE:
        raise RuntimeError(
            "O R09 mede o teste: defina RETINOPATIA_AVALIAR_TESTE=1 na "
            "primeira célula (e rode uma vez só)."
        )

    try:
        df_val, df_teste = preparar_imagens_avaliacao()

        modelo, resultado_validacao, resultado_teste = avaliar_modelo_final(
            df_val,
            df_teste
        )

        tabela_tflite = exportar_tflite(modelo, df_val)

        return resultado_validacao, resultado_teste, tabela_tflite

    finally:
        if desligar_ao_terminar:
            ambiente_colab.desligar_ambiente()
