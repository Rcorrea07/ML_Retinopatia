"""
Testes das regras de medida (`metricas`), com dados pequenos e respostas
conhecidas. Precisam de numpy, pandas e scikit-learn: pulam nesta máquina
e rodam no Colab (notebook Analisar_Resultados).

    python3 -m unittest discover -s tests -v
"""

import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

TEM_DEPENDENCIAS = all(
    importlib.util.find_spec(nome) is not None
    for nome in ("numpy", "pandas", "sklearn")
)


@unittest.skipUnless(TEM_DEPENDENCIAS, "numpy, pandas e scikit-learn ausentes")
class RegrasDeMedida(unittest.TestCase):

    def setUp(self):
        from retinopatia import metricas
        self.m = metricas

    def test_ponto_na_sensibilidade_tem_o_menor_falso_positivo(self):
        limiar, sensibilidade, especificidade = self.m.ponto_na_sensibilidade(
            [0, 0, 1, 1],
            [0.1, 0.4, 0.35, 0.8],
            0.9
        )

        self.assertAlmostEqual(limiar, 0.35)
        self.assertEqual(sensibilidade, 1.0)
        self.assertEqual(especificidade, 0.5)

    def test_sem_ponto_na_sensibilidade(self):
        self.assertEqual(
            self.m.ponto_na_sensibilidade([0, 1], [0.2, 0.8], 1.01),
            (None, None, None)
        )

    def test_paciente_doente_se_algum_olho_for(self):
        pacientes = ["b", "a", "a", "b"]
        classes = [0, 1, 0, 0]
        valores = [0.2, 0.9, 0.1, 0.4]

        maximo = self.m.por_paciente(pacientes, classes, valores)
        media = self.m.por_paciente(pacientes, classes, valores, "media")
        max_media = self.m.por_paciente(pacientes, classes, valores, "max_media")

        self.assertEqual(maximo.index.tolist(), ["a", "b"])
        self.assertEqual(maximo["target"].tolist(), [1, 0])
        self.assertEqual(maximo["valor"].tolist(), [0.9, 0.4])
        self.assertEqual(media["valor"].round(6).tolist(), [0.5, 0.3])
        self.assertEqual(max_media["valor"].round(6).tolist(), [0.7, 0.35])

        with self.assertRaises(ValueError):
            self.m.por_paciente(pacientes, classes, valores, "mediana")

    def test_kappa_arredonda_e_limita_o_grau(self):
        self.assertEqual(self.m.kappa_grau([0, 1, 2, 4], [-0.6, 1.2, 1.6, 4.7]), 1.0)
        self.assertLess(self.m.kappa_grau([0, 1, 2, 4], [4, 3, 1, 0]), 0)

    def test_metricas_no_limiar(self):
        resultado = self.m.metricas_no_limiar([0, 0, 1, 1], [0.1, 0.6, 0.7, 0.4], 0.5)

        self.assertEqual(resultado["sensibilidade"], 0.5)
        self.assertEqual(resultado["especificidade"], 0.5)
        self.assertEqual(resultado["roc_auc"], 0.75)

    def test_bootstrap_reproduzivel_e_em_volta_da_estimativa(self):
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(0)
        linhas = []

        for paciente in range(150):
            for olho in ("left", "right"):
                nivel = int(rng.choice([0, 0, 1, 2, 3]))
                linhas.append({
                    "patient_id": str(paciente),
                    "target": int(nivel > 0),
                    "prob_doente": float(np.clip(0.3 * nivel + rng.normal(0.2, 0.2), 0, 1))
                })

        tabela = pd.DataFrame(linhas)

        primeiro = self.m.intervalos_bootstrap(tabela, 0.4, 0.4, 200, seed=42)
        segundo = self.m.intervalos_bootstrap(tabela, 0.4, 0.4, 200, seed=42)

        pd.testing.assert_frame_equal(primeiro, segundo)

        cheio = self.m.metricas_no_limiar(tabela["target"], tabela["prob_doente"], 0.4)
        self.assertAlmostEqual(primeiro.loc[("olho", "roc_auc"), "estimativa"], cheio["roc_auc"])

        dentro = (
            (primeiro["ic95_inferior"] <= primeiro["estimativa"])
            & (primeiro["estimativa"] <= primeiro["ic95_superior"])
        )
        self.assertTrue(dentro.all())

    def test_diferenca_pareada_de_um_modelo_com_ele_mesmo_e_zero(self):
        import numpy as np

        gerador = np.random.default_rng(0)
        classes = gerador.integers(0, 2, 300)
        probabilidades = np.clip(0.3 * classes + gerador.normal(0.35, 0.2, 300), 0, 1)

        resultado = self.m.diferenca_pareada(classes, probabilidades, probabilidades, 0.9, 200, seed=1)

        self.assertTrue((resultado["diferenca"] == 0).all())
        self.assertTrue((resultado["ic95_inferior"] == 0).all())
        self.assertTrue((resultado["ic95_superior"] == 0).all())

    def test_diferenca_pareada_reconhece_o_modelo_melhor(self):
        import numpy as np

        gerador = np.random.default_rng(0)
        classes = gerador.integers(0, 2, 600)
        ruido = gerador.normal(0, 0.25, 600)

        bom = np.clip(0.5 * classes + 0.25 + 0.3 * ruido, 0, 1)
        fraco = np.clip(0.5 * classes + 0.25 + ruido, 0, 1)

        resultado = self.m.diferenca_pareada(classes, bom, fraco, 0.9, 300, seed=1)

        self.assertGreater(resultado.loc["roc_auc", "ic95_inferior"], 0)
        self.assertGreater(resultado.loc["especificidade", "ic95_inferior"], 0)
        self.assertAlmostEqual(
            resultado.loc["roc_auc", "diferenca"],
            resultado.loc["roc_auc", "a"] - resultado.loc["roc_auc", "b"]
        )


if __name__ == "__main__":
    unittest.main()
