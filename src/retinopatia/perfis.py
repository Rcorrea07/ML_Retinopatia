"""Leitura e identidade dos perfis, sem dependências de aprendizado de máquina."""

import hashlib
import json
import os
import re
from pathlib import Path


def hash_configuracao(valor):
    conteudo = json.dumps(valor, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(conteudo.encode()).hexdigest()


def carregar_perfil(nome=None):
    nome = nome or os.environ.get("RETINOPATIA_PERFIL", "")
    if not nome:
        return None
    caminho = Path(__file__).with_name("perfis_experimentos.json")
    perfis = json.loads(caminho.read_text(encoding="utf-8"))["perfis"]
    if nome not in perfis:
        raise ValueError(f"Perfil desconhecido: {nome}. Opções: {sorted(perfis)}")
    return {"id": nome, **perfis[nome]}


def validar_identificador(nome):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", nome):
        raise ValueError("Identificador deve conter apenas letras, números, _ e -.")
    return nome
