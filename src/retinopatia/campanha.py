"""Campanha Colab: processos isolados, ensaios, filas e avaliação sem teste."""

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from .perfis import carregar_perfil
from .persistencia import registrar_estado, salvar_json, sha256_arquivo


def _dados_ensaio():
    """Imagens sintéticas verificam o software; métricas não valem como score."""
    import cv2
    import numpy as np
    import pandas as pd
    from . import configuracao as c, pre_processamento
    pasta = Path("/content/retinopatia_ensaio") / c.ID_EXECUCAO
    pasta.mkdir(parents=True, exist_ok=True)
    linhas = []
    rng = np.random.default_rng(c.SEED_TREINO)
    for grau in range(c.NUMERO_GRAUS):
        for indice in range(c.AMOSTRAS_ENSAIO_POR_GRAU):
            imagem = np.zeros((512, 512, 3), dtype=np.uint8)
            disco = np.zeros((512, 512), dtype=np.uint8)
            cv2.circle(disco, (256, 256), 235, 255, -1)
            textura = rng.integers(25, 230, size=imagem.shape, dtype=np.uint8)
            imagem[disco > 0] = textura[disco > 0]
            processada, _, status = pre_processamento.processar_retina(imagem)
            if processada is None or status not in ("ok", c.STATUS_SUSPEITO):
                raise RuntimeError("Pré-processamento sintético falhou.")
            nome = f"sintetica_{grau}_{indice}_left.jpeg"
            if not cv2.imwrite(str(pasta / nome), processada):
                raise RuntimeError("Falha ao salvar imagem sintética.")
            linhas.append({"image": nome, "patient_id": f"sintetica_{grau}_{indice}",
                "level_original": grau, "target": int(grau > 0),
                "pasta_processada": str(pasta), "treino": indice % 2 == 0})
    dados = pd.DataFrame(linhas)
    treino, val = dados[dados.treino].copy(), dados[~dados.treino].copy()
    if c.DESTILACAO:
        treino["prob_professor"] = 0.1 + 0.8 * treino.target
    salvar_json(Path(c.PASTA_RESULTADOS) / "dados_ensaio.json",
                {"sinteticos": True, "nao_usar_como_score": True})
    return treino, val


def _preparar_dados():
    import pandas as pd
    from . import configuracao as c, ambiente_colab, conjunto_dados
    from .cache_experimentos import preparar_cache, auditar_duplicatas
    ambiente_colab.extrair_rotulos()
    dados, treino, val, teste = conjunto_dados.preparar_dataset(seed=c.SEED_SPLIT)
    hashes = [preparar_cache(dados)]
    if c.USAR_DADOS_EXTRA:
        extra = conjunto_dados.carregar_dados_extra(dados)
        hashes.append(preparar_cache(extra, extra=True))
        treino = conjunto_dados.ampliar_treino(treino, extra)
    else:
        treino["pasta_processada"] = c.PASTA_IMAGENS_OTIMIZADAS
    val["pasta_processada"] = c.PASTA_IMAGENS_OTIMIZADAS
    teste["pasta_processada"] = c.PASTA_IMAGENS_OTIMIZADAS
    treino = auditar_duplicatas(treino, val, teste, pd.concat(hashes, ignore_index=True))
    conjunto_dados.verificar_pacientes_disjuntos(treino, val, teste)
    originais = conjunto_dados._caminhos_splits(seed=c.SEED_SPLIT)
    salvar_json(Path(c.PASTA_RESULTADOS) / "manifesto_dados.json", {
        "splits": {nome: {"caminho": caminho, "sha256": sha256_arquivo(caminho)}
                   for nome, caminho in originais.items()},
        "rotulos_originais_sha256": sha256_arquivo(c.CAMINHO_CSV),
        "rotulos_extra_sha256": (sha256_arquivo(c.CAMINHO_ROTULOS_EXTRA)
                                  if c.USAR_DADOS_EXTRA else None),
        "numero_treino": len(treino), "numero_validacao": len(val),
        "numero_teste_historico": len(teste), "teste_predito": False})
    for nome, tabela in (("treino", treino), ("validacao", val)):
        tabela.to_csv(Path(c.PASTA_RESULTADOS) / f"dados_{nome}.csv", index=False)
    return treino, val


def _preparar_professor(treino):
    import numpy as np
    import tensorflow as tf
    from . import configuracao as c, pipeline_dados, avaliacao
    from .cache_experimentos import preparar_cache
    from .modelo import GeM
    if not Path(c.CAMINHO_PROFESSOR).is_file():
        raise FileNotFoundError(f"Professor ausente: {c.CAMINHO_PROFESSOR}")
    metadados = json.loads(Path(c.CAMINHO_CONFIGURACAO_PROFESSOR).read_text())
    if not (metadados.get("BACKBONE") == "EfficientNetB4"
            and list(metadados.get("TAMANHO_IMAGEM", [])) == [448, 448]
            and metadados.get("VERSAO_SPLIT") == "v2"
            and metadados.get("SEED") == c.SEED_SPLIT
            and metadados.get("USAR_DADOS_EXTRA") is True
            and str(metadados.get("COMMIT_CODIGO", "")).startswith("22f95c4")):
        raise ValueError("Proveniência do professor não corresponde ao treino/split esperado.")
    auditoria = json.loads((Path(c.PASTA_RESULTADOS) / "auditoria_dados.json").read_text())
    if auditoria["imagens_excluidas_treino"]:
        raise ValueError("Professor antigo pode ter visto cópias dos retidos; destilação recusada.")
    # Restaura os caches originais 448 do professor, sem passar o cache 384 ao modelo.
    from .cache_experimentos import _restaurar
    dados_prof = treino.copy()
    for extra, pasta_bruta in ((False, c.PASTA_IMAGENS_OTIMIZADAS),
                               (True, c.PASTA_IMAGENS_OTIMIZADAS_EXTRA)):
        parte = dados_prof[dados_prof.pasta_processada == pasta_bruta]
        pasta = Path("/content/dataset_professor_448" + ("_extra" if extra else ""))
        zip_origem = Path(c.PASTA_CACHE_DRIVE) / (
            "dataset_retina_v3_448" + ("_extra" if extra else "") + ".zip")
        _restaurar(zip_origem, pasta, parte.image.tolist())
        dados_prof.loc[parte.index, "pasta_processada"] = str(pasta)
    professor = tf.keras.models.load_model(c.CAMINHO_PROFESSOR,
                                          custom_objects={"GeM": GeM}, compile=False)
    professor.trainable = False
    ds = pipeline_dados.criar_dataset(dados_prof, treino=False, tamanho_imagem=(448, 448))
    prob, _ = avaliacao.prever_saidas(professor, ds, tta=False)
    if not np.isfinite(prob).all() or np.any((prob < 0) | (prob > 1)):
        raise ValueError("Probabilidades do professor inválidas.")
    treino = treino.copy()
    treino["prob_professor"] = prob
    treino[["image", "patient_id", "prob_professor"]].to_csv(
        Path(c.PASTA_RESULTADOS) / "alvos_professor_treino.csv", index=False)
    salvar_json(Path(c.PASTA_RESULTADOS) / "professor.json", {
        "modelo_sha256": sha256_arquivo(c.CAMINHO_PROFESSOR),
        "configuracao_sha256": sha256_arquivo(c.CAMINHO_CONFIGURACAO_PROFESSOR),
        "entrada": "v3 original 448; somente imagens do treino",
        "receita": "alvos offline sem TTA; aluno com augmentation; temperatura 1"})
    del professor, ds
    tf.keras.backend.clear_session()
    return treino


def executar_experimento():
    """Executa um perfil já selecionado no ambiente, com rastreabilidade."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from . import configuracao as c, utilitarios, pipeline_dados, treinamento, avaliacao, diagnostico
    from .modelo import construir_modelo, GeM
    from .persistencia import reservar_execucao

    novo = reservar_execucao(c)
    contador = 0
    def salvar_figuras(*args, **kwargs):
        nonlocal contador
        for numero in plt.get_fignums():
            contador += 1
            plt.figure(numero).savefig(Path(c.PASTA_RESULTADOS) / f"figura_{contador:03d}.png",
                                       dpi=140, bbox_inches="tight")
            plt.close(numero)
    plt.show = salvar_figuras
    try:
        if not tf.config.list_physical_devices("GPU"):
            raise RuntimeError("GPU ausente. Configure uma GPU antes de iniciar a campanha.")
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
        utilitarios.configurar_sementes(c.SEED_TREINO)
        utilitarios.configurar_precisao_mista()
        if novo:
            utilitarios.salvar_configuracao()
        registrar_estado(c.PASTA_RESULTADOS, "preparando_dados", perfil=c.PERFIL_EXPERIMENTO,
                         ensaio=c.ENSAIO, experimento=c.NOME_EXPERIMENTO)
        treino, val = _dados_ensaio() if c.ENSAIO else _preparar_dados()
        if c.DESTILACAO and not c.ENSAIO:
            treino = _preparar_professor(treino)
        treino_data = pipeline_dados.criar_dataset(treino, treino=True)
        val_data = pipeline_dados.criar_dataset(val, treino=False)
        if c.ENSAIO:
            treino_data = treino_data.take(c.PASSOS_ENSAIO)
        pipeline_dados.verificar_escala(treino_data, val_data)
        pesos = treinamento.calcular_pesos_classe(treino)
        salvar_json(Path(c.PASTA_RESULTADOS) / "pesos_classe.json", pesos)
        modelo, base = construir_modelo(mostrar_resumo=False)
        for fase in (1, 2):
            registrar_estado(c.PASTA_RESULTADOS, f"treinando_fase{fase}")
            modelo, base, historico = treinamento.treinar_fase_monitorada(
                modelo, base, treino_data, val_data, pesos, fase)
            # Mantém predições, curvas e limiares das duas fases, como no legado.
            resultado = avaliacao.avaliar_modelo(
                modelo, val_data, val, historico, f"fase_{fase}_sem_tta", tta=False)
            salvar_json(Path(c.PASTA_RESULTADOS) / f"metricas_validacao_fase{fase}.json",
                {k: v for k, v in resultado.items() if k not in ("tabela", "predicoes")})
        Path(c.PASTA_MODELOS).mkdir(parents=True, exist_ok=True)
        modelo.save(c.CAMINHO_MODELO_FINAL)
        # Recarregar verifica serialização de GeM, cabeças e formato do artefato.
        recarregado = tf.keras.models.load_model(c.CAMINHO_MODELO_FINAL,
                                                  custom_objects={"GeM": GeM}, compile=False)
        import numpy as np
        entrada = next(iter(val_data))[0]
        antes, depois = modelo(entrada, training=False), recarregado(entrada, training=False)
        for nome in antes:
            np.testing.assert_allclose(antes[nome].numpy(), depois[nome].numpy(), atol=1e-5, rtol=1e-4)
        del recarregado
        registrar_estado(c.PASTA_RESULTADOS, "avaliando_validacao")
        avaliacao.registrar_experimento(resultado)
        tabela = resultado["tabela"]
        diagnostico.relatorio_por_grau(tabela, resultado["limiar_olho"], nome="Validação")
        diagnostico.comparar_cabeca_grau(tabela)
        diagnostico.comparar_agregacao_paciente(tabela)
        if not c.ENSAIO:
            diagnostico.intervalo_bootstrap(tabela, resultado["limiar_olho"],
                resultado["limiar_paciente"], nome="Validação").to_csv(
                    Path(c.PASTA_RESULTADOS) / "bootstrap_validacao.csv")
            secundario = avaliacao.avaliar_modelo(
                modelo, val_data, val, None, "fase_2_tta4", tta=True)
            salvar_json(Path(c.PASTA_RESULTADOS) / "metricas_validacao_tta4.json",
                {k: v for k, v in secundario.items() if k not in ("tabela", "predicoes")})
        salvar_json(Path(c.PASTA_RESULTADOS) / "artefato_modelo.json", {
            "caminho": c.CAMINHO_MODELO_FINAL, "sha256": sha256_arquivo(c.CAMINHO_MODELO_FINAL),
            "recarga_verificada": True, "teste_avaliado": False})
        registrar_estado(c.PASTA_RESULTADOS, "concluido", modelo=c.CAMINHO_MODELO_FINAL)
        print(f"EXPERIMENTO CONCLUÍDO: {c.NOME_EXPERIMENTO}", flush=True)
    except BaseException as erro:
        registrar_estado(c.PASTA_RESULTADOS, "falhou", erro=f"{type(erro).__name__}: {erro}")
        raise


def executar_fila(perfis, posto="posto1", ensaio=False):
    """Uma GPU por notebook; cada perfil inicia um processo Python limpo."""
    from datetime import datetime, timezone
    for perfil in perfis:
        carregar_perfil(perfil)
    raiz = Path("/content/drive/MyDrive/ProjetoRetinopatia/campanhas/2026-10-20")
    pasta = raiz / ("ensaios" if ensaio else posto)
    pasta.mkdir(parents=True, exist_ok=True)
    versao = sha256_arquivo(Path(__file__).with_name("perfis_experimentos.json"))[:12]
    concluido = []
    for perfil in perfis:
        id_execucao = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
        ambiente = dict(os.environ, RETINOPATIA_PERFIL=perfil,
            RETINOPATIA_ID_EXECUCAO=id_execucao, RETINOPATIA_ENSAIO="1" if ensaio else "0",
            PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
        registro = pasta / f"{perfil}_{id_execucao}.json"
        salvar_json(registro, {"perfil": perfil, "id": id_execucao, "estado": "iniciando",
                               "versao_perfis": versao, "ensaio": ensaio})
        comando = [sys.executable, "-m", "retinopatia.campanha", "--perfil", perfil]
        print(f"INICIANDO {perfil} | {posto} | ensaio={ensaio}", flush=True)
        with (pasta / f"{perfil}_{id_execucao}.log").open("w") as log:
            processo = subprocess.Popen(comando, env=ambiente, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1)
            for linha in processo.stdout:
                print(linha, end="", flush=True)
                log.write(linha)
                log.flush()
            retorno = processo.wait()
        salvar_json(registro, {"perfil": perfil, "id": id_execucao,
            "estado": "concluido" if retorno == 0 else "falhou", "retorno": retorno,
            "versao_perfis": versao, "ensaio": ensaio})
        concluido.append({"perfil": perfil, "retorno": retorno})
        if ensaio and retorno:
            raise RuntimeError(f"Ensaio {perfil} falhou; verificar o log antes de treinar.")
    return concluido


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perfil", required=True)
    args = parser.parse_args()
    carregar_perfil(args.perfil)
    os.environ["RETINOPATIA_PERFIL"] = args.perfil
    executar_experimento()


if __name__ == "__main__":
    main()
