"""Verifica isolamento de perfis e persistência sem instalar TensorFlow."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from retinopatia.perfis import carregar_perfil, validar_identificador
from retinopatia.persistencia import reservar_execucao, salvar_json


class ConfiguracaoCampanha(unittest.TestCase):
    def resolver(self, perfil, seed=42):
        ambiente = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
                        RETINOPATIA_PERFIL=perfil, RETINOPATIA_SEED_TREINO=str(seed))
        comando = """from retinopatia import configuracao as c
import json
print(json.dumps({'split':c.SEED_SPLIT,'seed':c.SEED_TREINO,
'modelo':c.NOME_MODELO,'resultado':c.NOME_EXPERIMENTO,
'cache':c.PASTA_IMAGENS_OTIMIZADAS,'fundo':c.COR_FUNDO,
'grau':c.MODO_GRAU,'tta':c.TTA_ATIVO,'classes':c.NUMERO_GRAUS}))"""
        return json.loads(subprocess.check_output([sys.executable, "-c", comando],
                                                  env=ambiente, text=True))

    def test_semente_treino_nao_muda_split_ou_cache(self):
        a, b = self.resolver("E01"), self.resolver("E01", 17)
        self.assertEqual(a["split"], b["split"])
        self.assertEqual(a["cache"], b["cache"])
        self.assertNotEqual(a["modelo"], b["modelo"])

    def test_repeticao_nao_sobrescreve_e_perfis_compartilham_so_imagens_iguais(self):
        a, b = self.resolver("E01"), self.resolver("E01")
        self.assertNotEqual(a["resultado"], b["resultado"])
        self.assertEqual(a["cache"], self.resolver("E04")["cache"])
        for perfil in ("E02", "E03", "E06"):
            self.assertNotEqual(a["cache"], self.resolver(perfil)["cache"])

    def test_classes_reais_e_ensaio_separado(self):
        c = self.resolver("E11")
        self.assertEqual(c["classes"], 5)
        self.assertEqual(c["grau"], "categorico")
        self.assertFalse(c["tta"])
        self.assertEqual(self.resolver("E02")["fundo"], 0)

    def test_erro_em_perfil_ou_identificador_invalido(self):
        with self.assertRaises(ValueError):
            carregar_perfil("E99")
        for identificador in ("../modelo", "nome/arquivo", "", "nome com espaco"):
            with self.assertRaises(ValueError):
                validar_identificador(identificador)

    def test_reserva_preserva_configuracao_e_exige_retomada_compativel(self):
        with tempfile.TemporaryDirectory() as pasta:
            raiz = Path(pasta) / "resultado"
            c = SimpleNamespace(PASTA_RESULTADOS=str(raiz), RETOMAR=False,
                                HASH_CONFIGURACAO="abc", COMMIT_CODIGO="123")
            self.assertTrue(reservar_execucao(c))
            arquivo = raiz / "configuracao.json"
            salvar_json(arquivo, {"HASH_CONFIGURACAO": "abc", "COMMIT_CODIGO": "123"})
            original = arquivo.read_bytes()
            with self.assertRaises(FileExistsError):
                reservar_execucao(c)
            c.RETOMAR = True
            self.assertFalse(reservar_execucao(c))
            c.HASH_CONFIGURACAO = "diferente"
            with self.assertRaises(ValueError):
                reservar_execucao(c)
            self.assertEqual(original, arquivo.read_bytes())

    def test_falha_de_json_nao_substitui_arquivo_anterior(self):
        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / "estado.json"
            salvar_json(caminho, {"estado": "treinando", "epoca": 3})
            original = caminho.read_bytes()
            with self.assertRaises(ValueError):
                salvar_json(caminho, {"metrica": float("nan")})
            self.assertEqual(caminho.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
