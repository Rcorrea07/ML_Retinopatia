"""
Testes da escolha de experimentos, sem TensorFlow.

Cada caso importa `configuracao` num processo novo, com as variáveis de
ambiente do experimento, como o notebook faz no Colab.

    python3 -m unittest discover -s tests -v
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

PASTA_SRC = Path(__file__).resolve().parents[1] / "src"

sys.path.insert(0, str(PASTA_SRC))

from retinopatia import experimentos  # noqa: E402


def resolver(experimento, **ambiente_extra):
    """
    Importa `configuracao` com o experimento escolhido e devolve as
    constantes que importam para os testes.
    """

    ambiente = dict(
        os.environ,
        PYTHONPATH=str(PASTA_SRC),
        RETINOPATIA_EXPERIMENTO=experimento,
        RETINOPATIA_ID_EXECUCAO="20261022_1200",
        **ambiente_extra
    )

    comando = """
import json
from retinopatia import configuracao as c
print(json.dumps({
    "backbone": c.BACKBONE,
    "tamanho": c.TAMANHO_IMAGEM,
    "sigma": c.SIGMA_BEN,
    "erosao": c.EROSAO_MASCARA,
    "seed_split": c.SEED_SPLIT,
    "seed_treino": c.SEED_TREINO,
    "nome": c.NOME_EXPERIMENTO,
    "resultados": c.PASTA_RESULTADOS,
    "modelo": c.CAMINHO_MODELO_FINAL,
    "cache": c.PASTA_IMAGENS_OTIMIZADAS,
    "zips": c.CAMINHOS_ZIP_CACHE,
    "suspeitas": c.CAMINHO_LISTA_SUSPEITAS,
    "pares": c.MODELO_PARES,
    "destilacao": c.DESTILACAO,
    "teste": c.AVALIAR_TESTE,
    "epocas": [c.EPOCAS_FASE1, c.EPOCAS_FASE2],
    "modo_tta": c.MODO_TTA,
    "modelo_escolhido": c.CAMINHO_MODELO_ESCOLHIDO,
    "externos": c.DATASETS_EXTERNOS,
    "pastas_externas": {n: c.PASTAS_IMAGENS_EXTERNAS[n] for n in c.DATASETS_EXTERNOS},
    "zips_externos": {n: c.CAMINHOS_ZIP_CACHE_EXTERNOS[n] for n in c.DATASETS_EXTERNOS},
}))
"""

    resultado = subprocess.run(
        [sys.executable, "-c", comando],
        env=ambiente,
        capture_output=True,
        text=True
    )

    if resultado.returncode != 0:
        raise RuntimeError(resultado.stderr)

    return json.loads(resultado.stdout)


class EscolhaDeExperimento(unittest.TestCase):

    def test_padrao_mantem_receita_v3_de_448(self):
        c = resolver(experimentos.EXPERIMENTO_PADRAO)

        self.assertEqual(c["tamanho"], [448, 448])
        self.assertEqual(c["sigma"], 10)
        self.assertEqual(c["erosao"], 3)
        self.assertEqual(c["cache"], "/content/dataset_retina_v3_448")
        self.assertTrue(c["zips"][1].endswith(
            "dados_internos/cache/dataset_retina_v3_448.zip"
        ))
        self.assertTrue(c["suspeitas"].endswith("mascaras_suspeitas_v3.csv"))

    def test_todos_os_experimentos_resolvem_para_pastas_distintas(self):
        resolvidos = {
            nome: resolver(nome) for nome in experimentos.EXPERIMENTOS
        }

        for chave in ("nome", "resultados", "modelo"):
            valores = [c[chave] for c in resolvidos.values()]
            self.assertEqual(len(valores), len(set(valores)), chave)

        for nome, c in resolvidos.items():
            self.assertTrue(c["nome"].startswith(nome))

    def test_resolucao_maior_tem_cache_proprio_e_receita_escalada(self):
        c = resolver("R02_b0_640")

        self.assertEqual(c["backbone"], "EfficientNetB0")
        self.assertEqual(c["tamanho"], [640, 640])
        self.assertAlmostEqual(c["sigma"], 10 * 640 / 448)
        self.assertEqual(c["erosao"], 4)
        self.assertEqual(c["cache"], "/content/dataset_retina_v3_640")
        self.assertTrue(c["zips"][0].startswith(
            "/content/drive/MyDrive/ProjetoRetinopatia_caches/"
        ))
        self.assertTrue(c["suspeitas"].endswith("mascaras_suspeitas_v3_640.csv"))

    def test_semente_de_treino_nao_muda_split_nem_cache(self):
        a = resolver("R01_b0_512")
        b = resolver("R01_b0_512", RETINOPATIA_SEED_TREINO="17")

        self.assertEqual(a["seed_split"], b["seed_split"])
        self.assertEqual(a["cache"], b["cache"])
        self.assertEqual(b["seed_treino"], 17)
        self.assertNotEqual(a["resultados"], b["resultados"])

    def test_teste_fechado_por_padrao(self):
        self.assertFalse(resolver("R01_b0_512")["teste"])
        self.assertTrue(
            resolver("R01_b0_512", RETINOPATIA_AVALIAR_TESTE="1")["teste"]
        )

    def test_fumaca_usa_uma_epoca_e_pasta_propria(self):
        c = resolver("R04_b0_pares_640", RETINOPATIA_FUMACA="1")

        self.assertEqual(c["epocas"], [1, 1])
        self.assertTrue(c["resultados"].endswith("_fumaca"))
        self.assertTrue(c["modelo"].startswith(c["resultados"]))
        self.assertTrue(c["pares"])

    def test_final_mede_o_b0_768_escolhido_com_tta2(self):
        padrao = resolver(experimentos.EXPERIMENTO_PADRAO)
        final = resolver("R09_final_b0_768", RETINOPATIA_AVALIAR_TESTE="1")

        self.assertEqual(padrao["modo_tta"], "tta4")
        self.assertEqual(final["modo_tta"], "tta2")
        self.assertEqual(final["tamanho"], [768, 768])
        self.assertTrue(final["teste"])
        self.assertIn("R10_b0_768", final["modelo_escolhido"])

    def test_externos_tem_pasta_e_cache_proprios_e_nome_proprio(self):
        r10 = resolver("R10_b0_768")
        r12 = resolver("R12_b0_768_externos")

        self.assertEqual(r10["externos"], [])
        self.assertEqual(
            r12["externos"],
            ["DDR", "APTOS", "DeepDRiD", "IDRiD"]
        )
        self.assertEqual(r12["cache"], r10["cache"])
        self.assertEqual(
            r12["pastas_externas"]["DDR"],
            "/content/dataset_retina_v3_768_ddr"
        )
        self.assertEqual(
            r12["zips_externos"]["DeepDRiD"][0],
            "/content/drive/MyDrive/ProjetoRetinopatia_caches/"
            "dataset_retina_v3_768_deepdrid.zip"
        )
        self.assertIn("_extra_externos_", r12["nome"])
        self.assertNotIn("_externos", r10["nome"])

    def test_dataset_externo_desconhecido_e_recusado(self):
        comando = (
            "from retinopatia import experimentos\n"
            "experimentos.EXPERIMENTOS['_teste_externo'] = "
            "{'DATASETS_EXTERNOS': ['Messidor']}\n"
            "from retinopatia import configuracao\n"
        )

        resultado = subprocess.run(
            [sys.executable, "-c", comando],
            env=dict(
                os.environ,
                PYTHONPATH=str(PASTA_SRC),
                RETINOPATIA_EXPERIMENTO="_teste_externo"
            ),
            capture_output=True,
            text=True
        )

        self.assertNotEqual(resultado.returncode, 0)
        self.assertIn("Datasets externos desconhecidos", resultado.stderr)

    def test_experimento_desconhecido_e_recusado(self):
        with self.assertRaises(RuntimeError) as erro:
            resolver("nao_existe")

        self.assertIn("Experimento desconhecido", str(erro.exception))

    def test_constante_com_nome_errado_e_recusada(self):
        # Experimento criado só no processo filho, com um erro de digitação
        comando = (
            "from retinopatia import experimentos\n"
            "experimentos.EXPERIMENTOS['_teste_errado'] = "
            "{'TAMANHO_IMAGENS': (512, 512)}\n"
            "from retinopatia import configuracao\n"
        )

        resultado = subprocess.run(
            [sys.executable, "-c", comando],
            env=dict(
                os.environ,
                PYTHONPATH=str(PASTA_SRC),
                RETINOPATIA_EXPERIMENTO="_teste_errado"
            ),
            capture_output=True,
            text=True
        )

        self.assertNotEqual(resultado.returncode, 0)
        self.assertIn("TAMANHO_IMAGENS", resultado.stderr)


if __name__ == "__main__":
    unittest.main()
