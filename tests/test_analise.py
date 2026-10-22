"""
Testes das análises sobre predições salvas, com dados sintéticos.
Precisam de pandas e scikit-learn (no Colab já vêm instalados).

    python3 -m unittest discover -s tests -v
"""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

TEM_DEPENDENCIAS = all(
    importlib.util.find_spec(nome) is not None
    for nome in ("numpy", "pandas", "sklearn")
)


def tabela_sintetica(pacientes=200, seed=0):
    """
    Dois olhos por paciente; a probabilidade separa razoavelmente as
    classes e o grau acompanha o nível real com ruído.
    """

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    linhas = []

    for paciente in range(pacientes):
        for olho in ("left", "right"):
            nivel = int(rng.choice([0, 0, 0, 1, 2, 3, 4]))
            linhas.append({
                "image": f"{paciente}_{olho}",
                "patient_id": str(paciente),
                "level_original": nivel,
                "target": int(nivel > 0),
                "prob_doente": float(np.clip(0.25 * nivel + rng.normal(0.2, 0.15), 0, 1)),
                "grau_previsto": float(nivel + rng.normal(0, 0.5))
            })

    return pd.DataFrame(linhas)


@unittest.skipUnless(TEM_DEPENDENCIAS, "numpy, pandas e scikit-learn ausentes")
class Analise(unittest.TestCase):

    def setUp(self):
        from retinopatia import analise
        self.analise = analise

    def test_le_os_tres_formatos_de_predicoes(self):
        tabela = tabela_sintetica()

        with tempfile.TemporaryDirectory() as pasta:
            atual = os.path.join(pasta, "R01_atual")
            os.makedirs(atual)
            nova = tabela.copy()
            for modo in self.analise.MODOS:
                nova[f"prob_doente_{modo}"] = nova["prob_doente"]
                nova[f"grau_previsto_{modo}"] = nova["grau_previsto"]
            nova.to_csv(os.path.join(atual, "predicoes_validacao_fase_2.csv"), index=False)

            campanha = os.path.join(pasta, "E01_campanha")
            os.makedirs(campanha)
            for modo in ("sem_tta", "tta4"):
                tabela.to_csv(
                    os.path.join(campanha, f"predicoes_validacao_fase_2_{modo}.csv"),
                    index=False
                )

            antiga = os.path.join(pasta, "EfficientNetB4_antiga")
            os.makedirs(antiga)
            tabela.to_csv(os.path.join(antiga, "predicoes_validacao_fase_2.csv"), index=False)

            os.makedirs(os.path.join(pasta, "R02_x_fumaca"))

            execucoes = self.analise.listar_execucoes(pasta)

        self.assertEqual(set(execucoes), {"R01_atual", "E01_campanha", "EfficientNetB4_antiga"})
        self.assertEqual(set(execucoes["R01_atual"]), set(self.analise.MODOS))
        self.assertEqual(set(execucoes["E01_campanha"]), {"sem_tta", "tta4"})
        self.assertEqual(set(execucoes["EfficientNetB4_antiga"]), {"tta4"})

        comparativa = self.analise.tabela_comparativa(execucoes)
        self.assertEqual(len(comparativa), 3 + 2 + 1)
        self.assertTrue(comparativa.loc[comparativa.execucao == "E01_campanha", "referencia"].all())

    def test_metricas_e_combinacoes(self):
        tabela = self.analise._tabela_modo(tabela_sintetica(), "prob_doente", "grau_previsto")

        resultado = self.analise.metricas(tabela)
        self.assertGreater(resultado["paciente_roc_auc"], 0.8)
        self.assertGreater(self.analise.kappa_grau(tabela), 0.7)

        for comparacao in (
            self.analise.combinador_paciente(tabela),
            self.analise.misturar_cabecas(tabela)
        ):
            self.assertEqual(len(comparacao), 2)
            self.assertTrue(comparacao["roc_auc"].between(0, 1).all())

        media = self.analise.ensemble([tabela, tabela])
        self.assertTrue((media["prob"].values == tabela["prob"].values).all())


if __name__ == "__main__":
    unittest.main()
