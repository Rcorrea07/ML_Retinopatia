"""
Testes das tabelas de dados (rótulos, paciente, divisão, pares e fumaça),
com dados sintéticos. Precisam de numpy, pandas e scikit-learn: pulam
nesta máquina e rodam no Colab (notebook Analisar_Resultados).

    python3 -m unittest discover -s tests -v
"""

import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

TEM_DEPENDENCIAS = all(
    importlib.util.find_spec(nome) is not None
    for nome in ("numpy", "pandas", "sklearn")
)


def rotulos_sinteticos(pacientes=300, seed=0):
    """
    CSV de rótulos no formato do EyePACS: dois olhos por paciente
    (`<id>_left.jpeg`, `<id>_right.jpeg`) e o grau de 0 a 4.
    """

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)

    linhas = [
        {"image": f"{paciente}_{olho}.jpeg", "level": int(rng.choice([0, 0, 0, 1, 2, 3, 4]))}
        for paciente in range(10, 10 + pacientes)
        for olho in ("left", "right")
    ]

    return pd.DataFrame(linhas)


@unittest.skipUnless(TEM_DEPENDENCIAS, "numpy, pandas e scikit-learn ausentes")
class TabelasDeDados(unittest.TestCase):

    def setUp(self):
        from retinopatia import conjunto_dados
        self.cd = conjunto_dados

    def preparar(self, **kwargs):
        dados = rotulos_sinteticos(**kwargs)
        dados = self.cd.criar_classe_binaria(dados)
        return self.cd.extrair_identificador_paciente(dados)

    def test_classe_binaria_junta_os_graus_1_a_4(self):
        dados = self.preparar()

        self.assertTrue((dados["target"] == (dados["level_original"] > 0)).all())
        self.assertEqual(set(dados["level_original"]), {0, 1, 2, 3, 4})

    def test_grau_invalido_para_a_execucao(self):
        dados = rotulos_sinteticos(pacientes=5)
        dados.loc[0, "level"] = 7

        with self.assertRaises(ValueError):
            self.cd.criar_classe_binaria(dados)

    def test_paciente_sai_do_nome_da_imagem(self):
        dados = self.preparar(pacientes=3)

        self.assertEqual(
            dados["patient_id"].tolist(),
            ["10", "10", "11", "11", "12", "12"]
        )

    def test_divisao_por_paciente_sem_vazamento(self):
        dados = self.preparar()

        with redirect_stdout(io.StringIO()):
            treino, val, teste = self.cd.dividir_treino_validacao_teste(dados)

        conjuntos = [set(df["patient_id"]) for df in (treino, val, teste)]

        self.assertTrue(conjuntos[0].isdisjoint(conjuntos[1]))
        self.assertTrue(conjuntos[0].isdisjoint(conjuntos[2]))
        self.assertTrue(conjuntos[1].isdisjoint(conjuntos[2]))
        self.assertEqual(len(treino) + len(val) + len(teste), len(dados))

        # Uma dobra de 10 para teste e outra para validação
        self.assertAlmostEqual(len(teste) / len(dados), 0.1, delta=0.03)
        self.assertAlmostEqual(len(val) / len(dados), 0.1, delta=0.03)

        # Os dois olhos de cada paciente ficam juntos
        for df in (treino, val, teste):
            self.assertTrue((df.groupby("patient_id").size() == 2).all())

    def test_divisao_e_reproduzivel(self):
        dados = self.preparar()

        with redirect_stdout(io.StringIO()):
            primeira = self.cd.dividir_treino_validacao_teste(dados)
            segunda = self.cd.dividir_treino_validacao_teste(dados)

        for a, b in zip(primeira, segunda):
            self.assertEqual(a["image"].tolist(), b["image"].tolist())

    def test_par_e_o_outro_olho_e_a_ordem_nao_muda(self):
        dados = self.preparar(pacientes=4)

        # Paciente com uma só foto usa a própria imagem como par
        dados = dados[dados["image"] != "13_right.jpeg"].reset_index(drop=True)

        with redirect_stdout(io.StringIO()):
            com_par = self.cd.adicionar_par(dados)

        self.assertEqual(com_par["image"].tolist(), dados["image"].tolist())
        pares = dict(zip(com_par["image"], com_par["image_par"]))
        self.assertEqual(pares["10_left.jpeg"], "10_right.jpeg")
        self.assertEqual(pares["10_right.jpeg"], "10_left.jpeg")
        self.assertEqual(pares["13_left.jpeg"], "13_left.jpeg")

    def test_fumaca_reduz_a_pacientes_inteiros(self):
        dados = self.preparar()

        with redirect_stdout(io.StringIO()):
            treino, val, teste = self.cd.dividir_treino_validacao_teste(dados)
            reduzidos = self.cd.reduzir_para_fumaca(
                treino, val, teste, pacientes_por_classe=5, ativo=True
            )
            intactos = self.cd.reduzir_para_fumaca(treino, val, teste, ativo=False)

        for original, reduzido in zip((treino, val, teste), reduzidos):
            pacientes = reduzido.groupby("patient_id")

            self.assertLessEqual(pacientes.ngroups, 10)
            self.assertTrue((pacientes.size() == 2).all())
            self.assertTrue(set(reduzido["image"]) <= set(original["image"]))

        self.assertIs(intactos[0], treino)


if __name__ == "__main__":
    unittest.main()
