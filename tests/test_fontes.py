"""
Testes das fontes de imagens (`fontes`): o registro de cada experimento
e o preparo a partir de um cache. Precisam de pandas, scikit-learn,
OpenCV, matplotlib, tqdm e do `unzip`: pulam nesta máquina e rodam no
Colab (notebook Analisar_Resultados).

    python3 -m unittest discover -s tests -v
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

PASTA_SRC = Path(__file__).resolve().parents[1] / "src"

sys.path.insert(0, str(PASTA_SRC))

TEM_DEPENDENCIAS = all(
    importlib.util.find_spec(nome) is not None
    for nome in ("pandas", "sklearn", "cv2", "matplotlib", "tqdm")
) and shutil.which("unzip") is not None


def fontes_do_experimento(experimento):
    """
    Nome, papel, pasta e caches de cada fonte de treino do experimento,
    num processo novo (a configuração é fixada no import).
    """

    comando = (
        "import json\n"
        "from retinopatia import configuracao, fontes\n"
        "print(json.dumps([\n"
        "    [f.nome, f.papel, f.pasta_processada, list(f.caminhos_zip_cache)]\n"
        "    for f in fontes.fontes_de_treino()\n"
        "]))\n"
    )

    resultado = subprocess.run(
        [sys.executable, "-c", comando],
        env=dict(
            os.environ,
            PYTHONPATH=str(PASTA_SRC),
            RETINOPATIA_EXPERIMENTO=experimento,
            RETINOPATIA_ID_EXECUCAO="20261022_1200"
        ),
        capture_output=True,
        text=True
    )

    if resultado.returncode != 0:
        raise RuntimeError(resultado.stderr)

    return json.loads(resultado.stdout.strip().splitlines()[-1])


@unittest.skipUnless(TEM_DEPENDENCIAS, "pandas, scikit-learn, OpenCV, matplotlib, tqdm ou unzip ausentes")
class FontesDeImagens(unittest.TestCase):

    def test_registro_de_cada_experimento(self):
        r10 = fontes_do_experimento("R10_b0_768")
        r12 = fontes_do_experimento("R12_b0_768_externos")

        self.assertEqual([f[0] for f in r10], ["EyePACS_extra"])
        self.assertEqual(
            [f[0] for f in r12],
            ["EyePACS_extra", "DDR", "APTOS", "DeepDRiD", "IDRiD"]
        )
        self.assertTrue(all(f[1] == "treino" for f in r12))

        # Os nomes de pasta e de cache são os de antes da refatoração
        self.assertEqual(r12[1][2], "/content/dataset_retina_v3_768_ddr")
        self.assertEqual(
            r12[0][3][0],
            "/content/drive/MyDrive/ProjetoRetinopatia_caches/dataset_retina_v3_768_extra.zip"
        )

    def test_papel_desconhecido_e_recusado(self):
        from retinopatia import fontes

        with self.assertRaises(ValueError):
            fontes.FonteDados(
                nome="X",
                papel="teste",
                pasta_processada="/tmp/x",
                caminhos_zip_cache=("/tmp/x.zip",),
                caminho_lista_suspeitas="/tmp/x.csv",
                ler_tabela=None,
                extrair=None
            )

    def test_preparar_fonte_restaura_o_cache_e_tira_as_que_faltam(self):
        import pandas as pd

        from retinopatia import fontes

        with tempfile.TemporaryDirectory() as pasta:
            processadas = os.path.join(pasta, "imagens", "dataset_teste")
            cache = os.path.join(pasta, "dataset_teste.zip")

            with zipfile.ZipFile(cache, "w") as arquivo_zip:
                for nome in ("a.jpg", "b.jpg", "c.jpg"):
                    arquivo_zip.writestr(f"dataset_teste/{nome}", b"jpg")

            tabela = pd.DataFrame({"image": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]})

            def nao_extrair(_):
                raise AssertionError("com cache, nada é extraído")

            fonte = fontes.FonteDados(
                nome="teste",
                papel="treino",
                pasta_processada=processadas,
                caminhos_zip_cache=(cache,),
                caminho_lista_suspeitas=os.path.join(pasta, "suspeitas.csv"),
                ler_tabela=lambda dados: tabela,
                extrair=nao_extrair,
                fracao_maxima_falhas=0.5
            )

            resultado = fontes.preparar_fonte(fonte)

            self.assertEqual(resultado["image"].tolist(), ["a.jpg", "b.jpg", "c.jpg"])

            # Acima da fração aceita, o cache está incompleto
            estrita = fontes.FonteDados(**{**fonte.__dict__, "fracao_maxima_falhas": 0.0})

            with self.assertRaises(RuntimeError):
                fontes.preparar_fonte(estrita)


if __name__ == "__main__":
    unittest.main()
