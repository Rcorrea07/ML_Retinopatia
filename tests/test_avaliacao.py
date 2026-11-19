"""
Testes da camada de medida da avaliação (`tabela_predicoes`,
`comparar_modos`, `medir_predicoes`), com predições sintéticas e sem
TensorFlow. Precisam de numpy, pandas, scikit-learn, matplotlib e
seaborn: pulam nesta máquina e rodam no Colab (Analisar_Resultados).

    python3 -m unittest discover -s tests -v
"""

import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path

PASTA_SRC = Path(__file__).resolve().parents[1] / "src"

sys.path.insert(0, str(PASTA_SRC))

TEM_DEPENDENCIAS = all(
    importlib.util.find_spec(nome) is not None
    for nome in ("numpy", "pandas", "sklearn", "matplotlib", "seaborn")
)


def conjunto_sintetico(pacientes=200, seed=0):
    """
    Tabela de um conjunto (dois olhos por paciente) e as predições dos
    três modos de TTA, como as de `prever_variantes`.
    """

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)

    niveis = rng.choice([0, 0, 0, 1, 2, 3, 4], size=2 * pacientes)

    df = pd.DataFrame({
        "image": [f"{p}_{olho}.jpeg" for p in range(pacientes) for olho in ("left", "right")],
        "patient_id": [str(p) for p in range(pacientes) for _ in range(2)],
        "level_original": niveis,
        "target": (niveis > 0).astype(int)
    })

    base = np.clip(0.2 * niveis + rng.normal(0.2, 0.15, len(df)), 0, 1)

    variantes = {
        modo: (
            np.clip(base + rng.normal(0, ruido, len(df)), 0, 1),
            niveis + rng.normal(0, 0.5, len(df))
        )
        for modo, ruido in (("sem_tta", 0.06), ("tta2", 0.04), ("tta4", 0.02))
    }

    return df, variantes


@unittest.skipUnless(TEM_DEPENDENCIAS, "numpy, pandas, scikit-learn, matplotlib ou seaborn ausentes")
class MedidaDaAvaliacao(unittest.TestCase):

    def setUp(self):
        import matplotlib
        matplotlib.use("Agg")

        from retinopatia import avaliacao
        self.av = avaliacao

    def test_tabela_tem_o_modo_principal_e_todos_os_modos(self):
        df, variantes = conjunto_sintetico()

        tabela = self.av.tabela_predicoes(df, variantes, "tta2")

        self.assertEqual(
            list(tabela.columns[:6]),
            ["image", "patient_id", "level_original", "target", "prob_doente", "grau_previsto"]
        )
        self.assertTrue((tabela["prob_doente"] == variantes["tta2"][0]).all())
        self.assertTrue((tabela["prob_doente_tta4"] == variantes["tta4"][0]).all())

    def test_validacao_escolhe_e_teste_aplica_os_limiares(self):
        df, variantes = conjunto_sintetico()

        validacao = self.av.medir_predicoes(df, variantes, "tta2")

        self.assertTrue(validacao["limiares_escolhidos"])
        self.assertGreaterEqual(validacao["metricas_paciente"]["sensibilidade"], 0.9)
        self.assertIn("limiar", validacao["comparacao_tta"])

        aplicado = self.av.medir_predicoes(
            df,
            variantes,
            "tta2",
            limiares=self.av.limiares_da_validacao(validacao)
        )

        self.assertFalse(aplicado["limiares_escolhidos"])
        self.assertIn("limiar_validacao", aplicado["comparacao_tta"])

        # No mesmo conjunto, aplicar os limiares escolhidos dá as mesmas métricas
        for nivel in ("olho", "paciente"):
            for nome in ("roc_auc", "sensibilidade", "especificidade"):
                self.assertAlmostEqual(
                    aplicado[f"metricas_{nivel}"][nome],
                    validacao[f"metricas_{nivel}"][nome]
                )

        for modo in variantes:
            self.assertAlmostEqual(
                aplicado["comparacao_tta"].loc[modo, "especificidade"],
                validacao["comparacao_tta"].loc[modo, "especificidade"]
            )

    def test_conjunto_com_uma_classe_so_e_recusado(self):
        df, variantes = conjunto_sintetico()
        df = df.assign(target=0)

        with self.assertRaises(ValueError):
            self.av.medir_predicoes(df, variantes, "tta2", limiares={
                "olho": 0.5, "paciente": 0.5, "modos": {m: 0.5 for m in variantes}
            })

    def test_medir_nao_importa_tensorflow(self):
        comando = (
            "import sys\n"
            "import matplotlib\n"
            "matplotlib.use('Agg')\n"
            "sys.path.insert(0, %r)\n"
            "from tests.test_avaliacao import conjunto_sintetico\n"
            "from retinopatia import avaliacao\n"
            "df, variantes = conjunto_sintetico()\n"
            "avaliacao.medir_predicoes(df, variantes, 'tta2')\n"
            "print('tensorflow' in sys.modules)\n"
        ) % str(PASTA_SRC.parent)

        resultado = subprocess.run(
            [sys.executable, "-c", comando],
            env=dict(os.environ, PYTHONPATH=str(PASTA_SRC)),
            capture_output=True,
            text=True
        )

        self.assertEqual(resultado.returncode, 0, resultado.stderr)
        self.assertEqual(resultado.stdout.strip().splitlines()[-1], "False")


if __name__ == "__main__":
    unittest.main()
