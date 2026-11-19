"""
Retrato dos nomes e caminhos que a configuração monta, sem TensorFlow.

`retrato_configuracao.json` guarda, para alguns experimentos, os nomes
de execução e de modelo, as pastas de resultados, os caches, as listas
de suspeitas e os caminhos dos splits como eram antes da refatoração.
Execuções antigas, caches e modelos no Drive dependem desses nomes: uma
mudança aqui quebra a ligação com eles. O commit vira `<commit>`, porque
muda a cada versão do código.

    python3 -m unittest discover -s tests -v
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

PASTA_SRC = Path(__file__).resolve().parents[1] / "src"
CAMINHO_RETRATO = Path(__file__).resolve().parent / "retrato_configuracao.json"


def resolver(experimento, constantes, ambiente_extra):
    """
    Valores das `constantes` da configuração com o experimento escolhido
    ("padrao" usa o experimento padrão), com o commit trocado por
    `<commit>`.
    """

    ambiente = dict(
        os.environ,
        PYTHONPATH=str(PASTA_SRC),
        RETINOPATIA_ID_EXECUCAO="20261022_1200",
        **ambiente_extra
    )

    ambiente.pop("RETINOPATIA_EXPERIMENTO", None)

    if experimento != "padrao":
        ambiente["RETINOPATIA_EXPERIMENTO"] = experimento

    comando = (
        "import json\n"
        "from retinopatia import configuracao as c\n"
        f"valores = {{n: getattr(c, n) for n in {constantes!r}}}\n"
        "texto = json.dumps(valores, ensure_ascii=False)\n"
        "print(texto.replace('_' + c.COMMIT_CODIGO + '_', '_<commit>_'))\n"
    )

    resultado = subprocess.run(
        [sys.executable, "-c", comando],
        env=ambiente,
        capture_output=True,
        text=True
    )

    if resultado.returncode != 0:
        raise RuntimeError(resultado.stderr)

    return json.loads(resultado.stdout)


class RetratoDaConfiguracao(unittest.TestCase):

    def test_nomes_e_caminhos_nao_mudam(self):
        retrato = json.loads(CAMINHO_RETRATO.read_text(encoding="utf-8"))

        for caso, esperado in retrato.items():
            experimento = caso.split()[0]

            with self.subTest(caso=caso):
                obtido = resolver(
                    experimento,
                    list(esperado["valores"]),
                    esperado["ambiente"]
                )

                for nome, valor in esperado["valores"].items():
                    self.assertEqual(obtido[nome], valor, f"{caso}: {nome}")


if __name__ == "__main__":
    unittest.main()
