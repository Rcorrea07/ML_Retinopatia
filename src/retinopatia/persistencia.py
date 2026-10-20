"""Arquivos atômicos e identidade das execuções, sem dependências de ML."""

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path


def sha256_arquivo(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            h.update(bloco)
    return h.hexdigest()


def salvar_json(caminho, dados):
    """A publicação só ocorre depois de escrever e fechar o arquivo completo."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_name(caminho.name + "." + uuid.uuid4().hex + ".parcial")
    def converter(valor):
        if hasattr(valor, "tolist"):
            return valor.tolist()
        if hasattr(valor, "item"):
            return valor.item()
        if isinstance(valor, set):
            return sorted(valor)
        return str(valor)
    with temporario.open("w", encoding="utf-8") as arquivo:
        json.dump(dados, arquivo, indent=2, ensure_ascii=False, default=converter,
                  allow_nan=False)
    os.replace(temporario, caminho)


def registrar_estado(pasta, estado, **detalhes):
    caminho = Path(pasta) / "estado_execucao.json"
    atual = json.loads(caminho.read_text()) if caminho.exists() else {}
    atual.update(estado=estado, atualizado_em=datetime.now(timezone.utc).isoformat(),
                 **detalhes)
    salvar_json(caminho, atual)
    print(f"ESTADO: {estado}", flush=True)


def reservar_execucao(configuracao):
    """Repetições não reutilizam resultados sem solicitação de retomada."""
    c = configuracao
    pasta = Path(c.PASTA_RESULTADOS)
    existente = pasta / "configuracao.json"
    if existente.exists():
        if not c.RETOMAR:
            raise FileExistsError(f"Execução já existe: {pasta}. Use um ID novo.")
        salvo = json.loads(existente.read_text())
        if (salvo.get("HASH_CONFIGURACAO") != c.HASH_CONFIGURACAO
                or salvo.get("COMMIT_CODIGO") != c.COMMIT_CODIGO):
            raise ValueError("Retomada recusada: código ou configuração diferentes.")
        return False
    # Nome com UUID separa sessões; mkdir impede colisões no mesmo filesystem.
    pasta.mkdir(parents=True, exist_ok=False)
    return True
