"""
Testes dos leitores dos datasets externos, com zips pequenos que imitam
a organização dos arquivos baixados (conferida em 30/10/2026).

Precisam de pandas e openpyxl: fora do Colab são pulados.

    python3 -m unittest discover -s tests -v
"""

import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import openpyxl  # noqa: F401
    import pandas as pd

    from retinopatia import datasets_externos
except ImportError:
    pd = None


def montar_zip(arquivos):
    """
    Zip em memória: caminho -> conteúdo (bytes, ou DataFrame gravado
    como CSV ou planilha conforme a extensão).
    """

    memoria = io.BytesIO()

    with zipfile.ZipFile(memoria, "w") as arquivo_zip:
        for caminho, conteudo in arquivos.items():
            if isinstance(conteudo, pd.DataFrame):
                saida = io.BytesIO()

                if caminho.endswith(".csv"):
                    conteudo.to_csv(saida, index=False)
                else:
                    conteudo.to_excel(saida, index=False)

                conteudo = saida.getvalue()

            arquivo_zip.writestr(caminho, conteudo)

    memoria.seek(0)

    return zipfile.ZipFile(memoria)


@unittest.skipIf(pd is None, "pandas/openpyxl só no Colab")
class LeitoresDosDatasets(unittest.TestCase):

    def test_ddr_tira_a_extensao_e_poe_o_prefixo(self):
        arquivo = montar_zip({
            "DR_grading.csv": pd.DataFrame({
                "id_code": ["007-0004-000.jpg", "20170413.jpg"],
                "diagnosis": [0, 4],
            }),
            "DR_grading/DR_grading/007-0004-000.jpg": b"x",
            "DR_grading/DR_grading/20170413.jpg": b"x",
            "DR_grading/DR_grading/sem_rotulo.jpg": b"x",
        })

        tabela = datasets_externos.tabela_ddr(arquivo)

        self.assertEqual(
            tabela["image"].tolist(),
            ["ddr_007-0004-000.jpg", "ddr_20170413.jpg"]
        )
        self.assertEqual(
            tabela["membro"].tolist()[0],
            "DR_grading/DR_grading/007-0004-000.jpg"
        )
        self.assertEqual(tabela["level"].tolist(), [0, 4])

    def test_aptos_le_so_o_train_e_grava_jpg(self):
        arquivo = montar_zip({
            "train.csv": pd.DataFrame({
                "id_code": ["000c1434d8d7"], "diagnosis": [2],
            }),
            "test.csv": pd.DataFrame({"id_code": ["0005cfc8afb6"]}),
            "train_images/000c1434d8d7.png": b"x",
            "test_images/0005cfc8afb6.png": b"x",
        })

        tabela = datasets_externos.tabela_aptos(arquivo)

        self.assertEqual(tabela["image"].tolist(), ["aptos_000c1434d8d7.jpg"])
        self.assertEqual(
            tabela["membro"].tolist(),
            ["train_images/000c1434d8d7.png"]
        )

    def test_deepdrid_usa_o_grau_do_olho_e_so_fotos_aceitas(self):
        raiz = "DeepDRiD-master/regular_fundus_images/"

        def rotulos(pid):
            return pd.DataFrame({
                "patient_id": [pid] * 3,
                "image_id": [f"{pid}_l1", f"{pid}_l2", f"{pid}_r1"],
                "Overall quality": [1, 0, 1],
                "left_eye_DR_Level": [3.0, 3.0, None],
                "right_eye_DR_Level": [None, None, 1.0],
                "patient_DR_Level": [3, 3, 3],
            })

        arquivos = {
            f"{raiz}regular-fundus-training/regular-fundus-training.csv":
                rotulos(1),
            f"{raiz}regular-fundus-validation/regular-fundus-validation.csv":
                rotulos(1),
            f"{raiz}Online-Challenge1&2-Evaluation/Challenge1_labels.xlsx":
                pd.DataFrame({"image_id": ["347_l1"], "DR_Levels": [2]}),
            f"{raiz}Online-Challenge1&2-Evaluation/Challenge2_labels.xlsx":
                pd.DataFrame({
                    "image_id": ["347_l1"], "Overall quality": [1],
                }),
            f"{raiz}Online-Challenge1&2-Evaluation/Images/347/347_l1.jpg": b"x",
            "DeepDRiD-master/ultra-widefield_images/ultra-widefield-training/"
            "Images/1/1_l1.jpg": b"x",
        }

        for pasta in ("regular-fundus-training", "regular-fundus-validation"):
            for foto in ("1_l1", "1_l2", "1_r1"):
                arquivos[f"{raiz}{pasta}/Images/1/{foto}.jpg"] = b"x"

        tabela = datasets_externos.tabela_deepdrid(montar_zip(arquivos))

        self.assertEqual(
            tabela["image"].tolist(),
            [
                "deepdrid_treino_1_l1.jpg", "deepdrid_treino_1_r1.jpg",
                "deepdrid_validacao_1_l1.jpg", "deepdrid_validacao_1_r1.jpg",
                "deepdrid_avaliacao_347_l1.jpg",
            ]
        )
        self.assertEqual(tabela["level"].tolist(), [3, 1, 3, 1, 2])
        self.assertEqual(tabela["patient_id"].iloc[0], "deepdrid_treino_1")
        self.assertFalse(tabela["membro"].str.contains("ultra").any())

    def test_idrid_separa_os_conjuntos_de_nomes_repetidos(self):
        raiz = "B.%20Disease%20Grading/B. Disease Grading/"
        rotulos = pd.DataFrame({
            "Image name": ["IDRiD_001", None],
            "Retinopathy grade": [3, None],
            "Risk of macular edema ": [2, None],
        })

        arquivo = montar_zip({
            f"{raiz}2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv":
                rotulos,
            f"{raiz}2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv":
                rotulos.assign(**{"Retinopathy grade": [4, None]}),
            f"{raiz}1. Original Images/a. Training Set/IDRiD_001.jpg": b"x",
            f"{raiz}1. Original Images/b. Testing Set/IDRiD_001.jpg": b"x",
            "A.%20Segmentation/A. Segmentation/1. Original Images/"
            "a. Training Set/IDRiD_01.jpg": b"x",
        })

        tabela = datasets_externos.tabela_idrid(arquivo)

        self.assertEqual(
            tabela["image"].tolist(),
            ["idrid_training_IDRiD_001.jpg", "idrid_testing_IDRiD_001.jpg"]
        )
        self.assertEqual(tabela["level"].tolist(), [3, 4])
        self.assertTrue(tabela["membro"].iloc[1].endswith(
            "b. Testing Set/IDRiD_001.jpg"
        ))

    def test_rotulo_sem_foto_para_a_execucao(self):
        arquivo = montar_zip({
            "train.csv": pd.DataFrame({"id_code": ["falta"], "diagnosis": [1]}),
            "train_images/outra.png": b"x",
        })

        with self.assertRaises(ValueError):
            datasets_externos.tabela_aptos(arquivo)

    def test_manter_processadas_tolera_poucas_falhas(self):
        with tempfile.TemporaryDirectory() as pasta:
            for nome in ("a.jpg", "b.jpg"):
                Path(pasta, nome).write_bytes(b"x")

            tabela = pd.DataFrame({"image": ["a.jpg", "b.jpg", "c.jpg"]})

            mantidas = datasets_externos.manter_processadas(
                tabela, pasta, fracao_maxima=0.5
            )
            self.assertEqual(mantidas["image"].tolist(), ["a.jpg", "b.jpg"])

            with self.assertRaises(RuntimeError):
                datasets_externos.manter_processadas(
                    tabela, pasta, fracao_maxima=0.1
                )


if __name__ == "__main__":
    unittest.main()
