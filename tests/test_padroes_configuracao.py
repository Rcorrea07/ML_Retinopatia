"""
Regras de estrutura do pacote, conferidas sem TensorFlow.

- Parâmetro padrão que lê a `configuracao` (`def f(x=configuracao.X)`)
  guarda o valor do momento do import: o `%autoreload` e um segundo
  experimento no mesmo processo não o mudam. `LIMITE_POR_MODULO` é a
  contagem de 19/11/2026; ela só pode diminuir até zerar.
- O notebook de treinos paralelos não importa TensorFlow: `execucao`,
  `configuracao` e `experimentos` precisam importar sem ele.

    python3 -m unittest discover -s tests -v
"""

import ast
import os
import subprocess
import sys
import unittest
from pathlib import Path

PASTA_SRC = Path(__file__).resolve().parents[1] / "src"
PASTA_PACOTE = PASTA_SRC / "retinopatia"

# Parâmetros padrão que leem `configuracao`, por módulo (19/11/2026)
LIMITE_POR_MODULO = {
    "ambiente_colab": 18,
    "analise": 3,
    "avaliacao": 11,
    "conjunto_dados": 21,
    "datasets_externos": 2,
    "diagnostico": 15,
    "execucao": 2,
    "modelo": 11,
    "modelo_final": 7,
    "pipeline_dados": 12,
    "pre_processamento": 21,
    "treinamento": 15,
    "utilitarios": 4,
}


def padroes_da_configuracao(caminho):
    """
    Quantos parâmetros padrão do módulo leem um atributo de
    `configuracao`.
    """

    arvore = ast.parse(caminho.read_text(encoding="utf-8"))

    total = 0

    for no in ast.walk(arvore):
        if not isinstance(no, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue

        padroes = list(no.args.defaults) + [
            padrao for padrao in no.args.kw_defaults if padrao is not None
        ]

        total += sum(
            1
            for padrao in padroes
            for parte in ast.walk(padrao)
            if isinstance(parte, ast.Attribute)
            and isinstance(parte.value, ast.Name)
            and parte.value.id == "configuracao"
        )

    return total


class EstruturaDoPacote(unittest.TestCase):

    def test_padroes_da_configuracao_so_diminuem(self):
        for caminho in sorted(PASTA_PACOTE.glob("*.py")):
            modulo = caminho.stem

            with self.subTest(modulo=modulo):
                self.assertLessEqual(
                    padroes_da_configuracao(caminho),
                    LIMITE_POR_MODULO.get(modulo, 0),
                    f"{modulo}: leia a configuracao dentro da função "
                    "(parâmetro padrão None), não no padrão do parâmetro"
                )

    def test_modulos_do_notebook_paralelo_nao_importam_tensorflow(self):
        comando = (
            "import sys\n"
            "from retinopatia import configuracao, execucao, experimentos\n"
            "print('tensorflow' in sys.modules)\n"
        )

        resultado = subprocess.run(
            [sys.executable, "-c", comando],
            env=dict(os.environ, PYTHONPATH=str(PASTA_SRC)),
            capture_output=True,
            text=True
        )

        self.assertEqual(resultado.returncode, 0, resultado.stderr)
        self.assertEqual(resultado.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
