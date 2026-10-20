"""Caches verificáveis por receita, com reaproveitamento explícito do v3 legado."""

import hashlib
import binascii
import json
import os
import shutil
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import configuracao as c
from . import ambiente_colab, pre_processamento
from .perfis import hash_configuracao
from .persistencia import salvar_json, sha256_arquivo


def _nome_hash(nomes):
    return hash_configuracao(sorted(nomes))


def _equivale_v3():
    return (c.TAMANHO_IMAGEM == (448, 448) and c.APLICAR_BEN
            and c.SIGMA_BEN == 10 and c.METODO_MASCARA == "limiar_fixo"
            and c.BEN_RESPEITA_MASCARA and c.LIMIAR_MASCARA == 10
            and c.EROSAO_MASCARA == 3 and c.RAIO_MASCARA == 0.9
            and c.QUALIDADE_JPEG == 95 and c.COR_FUNDO == 128)


def _restaurar(zip_origem, pasta, nomes):
    """Copia só membros esperados, sem confiar em caminhos internos do ZIP."""
    pasta.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_origem) as arquivo:
        membros = {}
        esperados = set(nomes)
        for item in arquivo.infolist():
            nome = Path(item.filename).name
            if not item.is_dir() and nome in esperados:
                if nome in membros:
                    raise ValueError(f"Imagem duplicada no ZIP: {nome}")
                membros[nome] = item
        if set(membros) != esperados:
            raise ValueError(f"Cache incompleto: {len(esperados - set(membros))} imagens faltando.")
        for nome, item in membros.items():
            destino = pasta / nome
            if destino.exists() and destino.stat().st_size == item.file_size:
                crc = 0
                with destino.open("rb") as existente:
                    for bloco in iter(lambda: existente.read(1024 * 1024), b""):
                        crc = binascii.crc32(bloco, crc)
                if crc & 0xffffffff == item.CRC:
                    continue
            temporario = destino.with_suffix(".parcial")
            with arquivo.open(item) as origem, temporario.open("wb") as saida:
                shutil.copyfileobj(origem, saida)
            os.replace(temporario, destino)


def _inspecionar_imagem(item):
    nome, pasta = item
    caminho = pasta / nome
    conteudo = caminho.read_bytes()
    imagem = cv2.imdecode(np.frombuffer(conteudo, np.uint8), cv2.IMREAD_COLOR)
    if imagem is None or imagem.shape != c.FORMATO_ENTRADA:
        raise ValueError(f"Imagem inválida ou resolução errada: {caminho}")
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    miniatura = cv2.resize(cinza, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(miniatura.astype(np.float32))[:8, :8]
    bits = dct > np.median(dct.ravel()[1:])
    phash = np.packbits(bits).tobytes().hex()
    return {"image": nome, "bytes": len(conteudo),
            "sha256_jpeg": hashlib.sha256(conteudo).hexdigest(),
            "sha256_pixels": hashlib.sha256(imagem.tobytes()).hexdigest(),
            "phash": phash}


def _registrar_cache_na_execucao(pasta, extra, zip_origem=None):
    """Preserva manifesto e hashes no Drive mesmo sem publicar um ZIP novo."""
    sufixo = "extra" if extra else "treino"
    destino = Path(c.PASTA_RESULTADOS)
    destino.mkdir(parents=True, exist_ok=True)
    for origem, nome in ((pasta / "hashes_imagens.csv", f"hashes_cache_{sufixo}.csv"),
                          (pasta / "manifesto_cache.json", f"manifesto_cache_{sufixo}.json")):
        temporario = destino / (nome + "." + uuid.uuid4().hex + ".parcial")
        shutil.copyfile(origem, temporario)
        os.replace(temporario, destino / nome)
    salvar_json(destino / f"armazenamento_cache_{sufixo}.json", {
        "pasta_local": str(pasta), "zip_reutilizado": zip_origem,
        "persistir_caches_novos": c.PERSISTIR_CACHES_NOVOS,
        "imagens_locais_volateis": True,
        "retomada": "Reutiliza ZIP existente ou refaz as imagens após perder o runtime."})


def preparar_cache(dados, extra=False):
    """Restaura ou constrói uma receita; valida cada imagem antes de treinar."""
    pasta = Path(c.PASTA_IMAGENS_OTIMIZADAS_EXTRA if extra else c.PASTA_IMAGENS_OTIMIZADAS)
    zip_destino = Path(c.CAMINHO_ZIP_CACHE_EXTRA if extra else c.CAMINHO_ZIP_CACHE)
    nomes = dados["image"].tolist()
    nomes_hash = _nome_hash(nomes)
    manifesto = pasta / "manifesto_cache.json"
    if manifesto.exists():
        m = json.loads(manifesto.read_text())
        if m["hash_receita"] != c.HASH_PRE_PROCESSAMENTO or m["hash_nomes"] != nomes_hash:
            raise ValueError("Manifesto de cache incompatível.")
        hashes = pasta / "hashes_imagens.csv"
        if (hashes.exists() and sha256_arquivo(hashes) == m["hash_tabela"]
                and all((pasta / n).is_file() for n in nomes)):
            _registrar_cache_na_execucao(pasta, extra, m.get("zip_disponivel"))
            return pd.read_csv(hashes, dtype={"phash": str})
    origem = None
    if zip_destino.exists():
        origem = str(zip_destino)
    elif _equivale_v3():
        legado = Path(c.PASTA_CACHE_DRIVE) / (
            "dataset_retina_v3_448" + ("_extra" if extra else "") + ".zip")
        if legado.exists():
            origem = str(legado)
    if origem:
        print(f"Restaurando e validando {len(nomes)} imagens: {origem}", flush=True)
        _restaurar(origem, pasta, nomes)
    else:
        nome = "test" if extra else "train"
        brutas = c.PASTA_IMAGENS_EXTRA if extra else c.PASTA_IMAGENS
        criada_agora = not Path(brutas).exists()
        ambiente_colab._extrair_imagens(nome, brutas, nomes_esperados=nomes)
        pre_processamento.pre_processar_dataset(
            dados, pasta_imagens=brutas, pasta_saida=str(pasta),
            caminho_lista_suspeitas=(c.CAMINHO_LISTA_SUSPEITAS_EXTRA if extra
                                    else c.CAMINHO_LISTA_SUSPEITAS))
        origem = "processamento_das_fotos_originais"
        # Só remove a extração temporária criada por esta chamada no /content.
        # O ZIP e os caches do Drive permanecem preservados.
        if criada_agora and Path(brutas).parent == Path(c.PASTA_BASE):
            shutil.rmtree(brutas)
    with ThreadPoolExecutor(max_workers=min(c.MAXIMO_WORKERS, 4)) as executor:
        tabela = pd.DataFrame(executor.map(_inspecionar_imagem,
                                          ((n, pasta) for n in nomes)))
    tabela.to_csv(pasta / "hashes_imagens.csv", index=False)
    # Um ZIP legado equivalente já é uma cópia persistente suficiente.
    # Nunca o duplica sob o nome com hash da nova campanha.
    zip_disponivel = origem if origem != "processamento_das_fotos_originais" else None
    salvar_json(manifesto, {"hash_receita": c.HASH_PRE_PROCESSAMENTO,
        "hash_nomes": nomes_hash, "numero_imagens": len(nomes),
        "formato": c.FORMATO_ENTRADA, "origem": origem,
        "zip_disponivel": zip_disponivel,
        "hash_tabela": sha256_arquivo(pasta / "hashes_imagens.csv"),
        "tipo_hash": "JPEG e pixels processados; nao substituem auditoria das fotos brutas"})
    if not zip_disponivel and c.PERSISTIR_CACHES_NOVOS and not zip_destino.exists():
        # Cada sessão usa um temporário próprio; só publica um ZIP completo.
        temporario = zip_destino.with_name(zip_destino.name + "." + uuid.uuid4().hex + ".parcial")
        zip_local = Path("/content") / (uuid.uuid4().hex + ".zip")
        with zipfile.ZipFile(zip_local, "w", compression=zipfile.ZIP_STORED) as arquivo:
            for nome in [*nomes, "manifesto_cache.json", "hashes_imagens.csv"]:
                arquivo.write(pasta / nome, arcname=pasta.name + "/" + nome)
        zip_destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(zip_local, temporario)
        os.replace(temporario, zip_destino)
        zip_local.unlink()
        zip_disponivel = str(zip_destino)
        metadados = json.loads(manifesto.read_text())
        metadados["zip_disponivel"] = zip_disponivel
        salvar_json(manifesto, metadados)
    elif not zip_disponivel:
        print("Cache novo mantido no /content; manifesto e hashes serão salvos "
              "com os resultados no Drive.", flush=True)
    _registrar_cache_na_execucao(pasta, extra, zip_disponivel)
    return tabela


def auditar_duplicatas(treino, validacao, teste, hashes):
    """Retira do treino cópias exatas dos retidos; registra suspeitas por pHash."""
    todos = pd.concat([treino.assign(conjunto="treino"),
                       validacao.assign(conjunto="validacao"),
                       teste.assign(conjunto="teste")], ignore_index=True)
    if todos["image"].duplicated().any():
        raise ValueError("Nomes de imagens repetidos após juntar dados extras.")
    tabela = todos.merge(hashes, on="image", how="left", validate="one_to_one")
    if tabela["sha256_pixels"].isna().any():
        raise ValueError("Auditoria incompleta: há imagens sem hash.")
    retidos = set(tabela.loc[tabela.conjunto != "treino", "sha256_pixels"])
    excluir = tabela[(tabela.conjunto == "treino") & tabela.sha256_pixels.isin(retidos)]
    pasta = Path(c.PASTA_RESULTADOS)
    excluir.to_csv(pasta / "duplicatas_exatas_excluidas_treino.csv", index=False)
    # Correspondência perceptual exata é apenas suspeita, sem mudar rótulos.
    suspeitas = tabela[tabela.groupby("phash")["conjunto"].transform("nunique") > 1]
    suspeitas.to_csv(pasta / "duplicatas_perceptuais_suspeitas.csv", index=False)
    conflitos = tabela[tabela.groupby("sha256_pixels")["level_original"].transform("nunique") > 1]
    conflitos.to_csv(pasta / "conflitos_rotulo_pixels.csv", index=False)
    salvar_json(pasta / "auditoria_dados.json", {
        "imagens_excluidas_treino": len(excluir), "suspeitas_phash": len(suspeitas),
        "conflitos_rotulo": len(conflitos),
        "limite": "Hashes de imagens processadas; semelhanca aproximada e fotos brutas nao atestadas.",
        "teste_historico_ja_consultado": True})
    print(f"Auditoria: {len(excluir)} cópias exatas retiradas do treino; "
          f"{len(suspeitas)} suspeitas perceptuais registradas.", flush=True)
    return treino[~treino.image.isin(excluir.image)].copy()
