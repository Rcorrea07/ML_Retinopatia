"""
Testes da validação externa (`validacao_externa`), sem GPU: as travas
dos limiares do R09 e a medida completa de uma base, com predições
sintéticas. Rodam num processo com o experimento V01_r12_messidor2 (a
configuração é fixada no import). Precisam de numpy, pandas,
scikit-learn, matplotlib, seaborn, OpenCV e tqdm: pulam nesta máquina e
rodam no Colab (notebook Analisar_Resultados).

    python3 -m unittest discover -s tests -v
"""

import importlib.util
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

PASTA_SRC = Path(__file__).resolve().parents[1] / "src"

TEM_DEPENDENCIAS = all(
    importlib.util.find_spec(nome) is not None
    for nome in ("numpy", "pandas", "sklearn", "matplotlib", "seaborn", "cv2", "tqdm")
)

# Monta um conjunto sintético (dois olhos por paciente) e as predições
# dos três modos de TTA, como as de `prever_variantes`
PREPARO = textwrap.dedent("""
    import os
    import tempfile

    import matplotlib
    matplotlib.use("Agg")

    import numpy as np
    import pandas as pd

    from retinopatia import avaliacao, configuracao, validacao_externa

    def conjunto(pacientes, seed):
        rng = np.random.default_rng(seed)
        niveis = rng.choice([0, 0, 0, 1, 2, 3, 4], size=2 * pacientes)
        df = pd.DataFrame({
            "image": [f"{p}_{o}.jpg" for p in range(pacientes) for o in ("l", "r")],
            "patient_id": [str(p) for p in range(pacientes) for _ in range(2)],
            "level_original": niveis,
            "target": (niveis > 0).astype(int)
        })
        base = np.clip(0.2 * niveis + rng.normal(0.2, 0.15, len(df)), 0, 1)
        variantes = {
            modo: (np.clip(base + rng.normal(0, r, len(df)), 0, 1), niveis + rng.normal(0, 0.5, len(df)))
            for modo, r in (("sem_tta", 0.06), ("tta2", 0.04), ("tta4", 0.02))
        }
        return df, variantes

    pasta_r09 = tempfile.mkdtemp()
    df_val, variantes_val = conjunto(200, 0)
    tabela_val = avaliacao.tabela_predicoes(df_val, variantes_val, "tta2")
    tabela_val.to_csv(os.path.join(pasta_r09, "predicoes_validacao_modelo_final.csv"), index=False)
    medido = avaliacao.medir_predicoes(df_val, variantes_val, "tta2")

    def gravar_resumo(**mudancas):
        linha = {
            "modelo_avaliado": os.path.basename(configuracao.CAMINHO_MODELO_ESCOLHIDO),
            "modo_tta": "tta2",
            "limiar_olho": medido["limiar_olho"],
            "limiar_paciente": medido["limiar_paciente"],
            **mudancas
        }
        pd.DataFrame([linha]).to_csv(os.path.join(pasta_r09, "resumo.csv"), index=False)
""")


def rodar(codigo):
    """
    Roda `PREPARO` + `codigo` num processo com o experimento
    V01_r12_messidor2 e devolve a última linha impressa.
    """

    resultado = subprocess.run(
        [sys.executable, "-c", PREPARO + textwrap.dedent(codigo)],
        env=dict(
            os.environ,
            PYTHONPATH=str(PASTA_SRC),
            RETINOPATIA_EXPERIMENTO="V01_r12_messidor2",
            RETINOPATIA_ID_EXECUCAO="20261022_1200"
        ),
        capture_output=True,
        text=True
    )

    if resultado.returncode != 0:
        raise AssertionError(resultado.stderr[-3000:])

    return resultado.stdout.strip().splitlines()[-1]


@unittest.skipUnless(TEM_DEPENDENCIAS, "dependências da avaliação ausentes")
class ValidacaoExterna(unittest.TestCase):

    def test_limiares_conferidos_com_o_r09(self):
        saida = rodar("""
            gravar_resumo()
            limiares, tabela = validacao_externa.limiares_do_modelo_final(pasta_r09)
            assert abs(limiares["olho"] - medido["limiar_olho"]) < 1e-12
            assert set(limiares["modos"].index) == {"sem_tta", "tta2", "tta4"}
            assert len(tabela) == len(df_val)

            recusas = 0
            for mudanca in (
                {"modelo_avaliado": "outro.keras"},
                {"modo_tta": "tta4"},
                {"limiar_olho": medido["limiar_olho"] + 0.01},
            ):
                gravar_resumo(**mudanca)
                try:
                    validacao_externa.limiares_do_modelo_final(pasta_r09)
                except RuntimeError:
                    recusas += 1
            print(recusas)
        """)

        self.assertEqual(saida, "3")

    def test_medida_de_uma_base_com_desfecho_principal(self):
        saida = rodar("""
            gravar_resumo()
            limiares, tabela_validacao = validacao_externa.limiares_do_modelo_final(pasta_r09)

            # Base externa: 150 pacientes; 10 deles com um só olho graduável
            df, variantes = conjunto(150, 1)
            fica = np.ones(len(df), dtype=bool)
            fica[1:20:2] = False
            df = df[fica].reset_index(drop=True)
            variantes = {m: (p[fica], g[fica]) for m, (p, g) in variantes.items()}

            pasta = tempfile.mkdtemp()
            partes = validacao_externa.medir_base(
                "Base", df, variantes, limiares, tabela_validacao, nao_graduaveis=10, pasta=pasta
            )

            resumo = pd.read_csv(os.path.join(pasta, "resumo_externo.csv")).iloc[0]
            assert resumo["pacientes"] == 150 and resumo["pacientes_dois_olhos"] == 140
            assert resumo["fotos_nao_graduaveis"] == 10
            assert 0 <= resumo["principal_paciente_roc_auc_ic95_inferior"] <= resumo["principal_paciente_roc_auc"]
            assert abs(resumo["limiar_paciente"] - limiares["paciente"]) < 1e-12
            assert not partes["avaliacao"]["limiares_escolhidos"]
            assert os.path.isfile(os.path.join(pasta, "predicoes_base.csv"))
            assert os.path.isfile(os.path.join(pasta, "validacao_externa_base_referenciavel.csv"))
            print("ok")
        """)

        self.assertEqual(saida, "ok")


if __name__ == "__main__":
    unittest.main()
