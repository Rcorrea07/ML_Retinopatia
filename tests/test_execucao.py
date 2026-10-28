"""
Testes dos treinos simultâneos numa sessão (execucao), sem TensorFlow.

Como em test_experimentos, cada caso importa o pacote num processo novo,
com as variáveis de ambiente que o notebook definiria.

    python3 -m unittest discover -s tests -v
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

PASTA_SRC = Path(__file__).resolve().parents[1] / "src"


def rodar(comando, **ambiente_extra):
    """
    Executa `comando` (Python) com o pacote importável e devolve o JSON
    que ele imprime.
    """

    ambiente = dict(
        os.environ,
        PYTHONPATH=str(PASTA_SRC),
        RETINOPATIA_EXPERIMENTO="R10_b0_768",
        RETINOPATIA_ID_EXECUCAO="20261028_1200"
    )
    ambiente.update(ambiente_extra)

    resultado = subprocess.run(
        [sys.executable, "-c", comando],
        env=ambiente,
        capture_output=True,
        text=True
    )

    if resultado.returncode != 0:
        raise RuntimeError(resultado.stderr)

    return json.loads(resultado.stdout)


class TreinosSimultaneos(unittest.TestCase):

    def test_quantos_cabem_pela_memoria(self):
        r = rodar(
            "import json\n"
            "from retinopatia.execucao import simultaneos_possiveis as s\n"
            "print(json.dumps({\n"
            "    'a100_80': s(3, ram_gb=167, vram_gb=80),\n"
            "    'a100_40': s(3, ram_gb=83, vram_gb=40),\n"
            "    'l4': s(3, ram_gb=53, vram_gb=22),\n"
            "    'pouca_memoria': s(3, ram_gb=12, vram_gb=15),\n"
            "    'um_treino': s(1, ram_gb=167, vram_gb=80),\n"
            "}))\n"
        )

        self.assertEqual(r["a100_80"], 3)
        self.assertEqual(r["a100_40"], 2)
        self.assertEqual(r["l4"], 1)
        self.assertEqual(r["pouca_memoria"], 1)
        self.assertEqual(r["um_treino"], 1)

    def test_ambiente_de_cada_treino(self):
        r = rodar(
            "import json\n"
            "from retinopatia.execucao import ambiente_do_treino, PASTA_SRC\n"
            "a = ambiente_do_treino('R11_v2b0_768', 17, False)\n"
            "f = ambiente_do_treino('R11_v2b0_768', 42, True)\n"
            "print(json.dumps({'a': a, 'f': f, 'src': PASTA_SRC}))\n"
        )

        a, f = r["a"], r["f"]

        self.assertEqual(a["RETINOPATIA_EXPERIMENTO"], "R11_v2b0_768")
        self.assertEqual(a["RETINOPATIA_SEED_TREINO"], "17")
        self.assertEqual(a["RETINOPATIA_FUMACA"], "0")
        self.assertEqual(f["RETINOPATIA_FUMACA"], "1")
        self.assertEqual(a["RETINOPATIA_AVALIAR_TESTE"], "0")
        self.assertEqual(a["TF_FORCE_GPU_ALLOW_GROWTH"], "true")
        self.assertEqual(a["MPLBACKEND"], "Agg")
        self.assertTrue(a["PYTHONPATH"].startswith(r["src"]))
        # Todos os treinos da sessão herdam o ID do notebook
        self.assertEqual(a["RETINOPATIA_ID_EXECUCAO"], "20261028_1200")

    def test_mesma_sessao_nao_sobrescreve_modelos(self):
        comando = (
            "import json\n"
            "from retinopatia import configuracao as c\n"
            "print(json.dumps({'modelo': c.CAMINHO_MODELO_FINAL,\n"
            "    'resultados': c.PASTA_RESULTADOS,\n"
            "    'pesos': c.CAMINHO_PESOS_FASE2}))\n"
        )

        semente_42 = rodar(comando, RETINOPATIA_EXPERIMENTO="R11_v2b0_768")
        semente_17 = rodar(
            comando,
            RETINOPATIA_EXPERIMENTO="R11_v2b0_768",
            RETINOPATIA_SEED_TREINO="17"
        )

        for chave in ("modelo", "resultados", "pesos"):
            self.assertNotEqual(semente_42[chave], semente_17[chave], chave)

        # A semente padrão mantém o nome antigo do modelo
        self.assertNotIn("seed42", semente_42["modelo"])
        self.assertIn("_seed17_", semente_17["modelo"])

        # Os checkpoints ficam no disco local, fora do Drive
        self.assertTrue(semente_17["pesos"].startswith("/content/checkpoints/"))


if __name__ == "__main__":
    unittest.main()
