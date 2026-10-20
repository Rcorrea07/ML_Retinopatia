"""Verificação de armazenamento com imagens sintéticas; executar só no Colab.

RETINOPATIA_PERFIL=E01 python tests/colab_verificar_cache.py
Não monta o Drive, não lê fotos reais e não treina modelos.
"""

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

from retinopatia import configuracao as c, pre_processamento
from retinopatia.cache_experimentos import preparar_cache


class ArmazenamentoCache(unittest.TestCase):
    def test_reuso_legado_e_receitas_locais_preservam_metadados(self):
        with tempfile.TemporaryDirectory() as temporaria:
            raiz = Path(temporaria)
            brutas, cache, local = (raiz / n for n in ("brutas", "drive_cache", "local"))
            brutas.mkdir()
            cache.mkdir()
            nomes = ["sintetica_0.jpeg", "sintetica_1.jpeg"]
            dados = pd.DataFrame({"image": nomes})
            legado = cache / "dataset_retina_v3_448.zip"
            with zipfile.ZipFile(legado, "w") as arquivo:
                for i, nome in enumerate(nomes):
                    imagem = np.zeros((512, 512, 3), dtype=np.uint8)
                    cv2.circle(imagem, (256, 256), 230, (70 + i * 25, 120, 160), -1)
                    self.assertTrue(cv2.imwrite(str(brutas / nome), imagem))
                    processada, _, _ = pre_processamento.processar_retina(imagem)
                    ok, jpeg = cv2.imencode(".jpeg", processada)
                    self.assertTrue(ok)
                    arquivo.writestr("cache/" + nome, jpeg.tobytes())
            legado_original = legado.read_bytes()
            destino_zip = cache / "receita_nova.zip"
            opcoes = dict(PASTA_IMAGENS_OTIMIZADAS=str(local),
                PASTA_IMAGENS=str(brutas), CAMINHO_ZIP_CACHE=str(destino_zip),
                PASTA_CACHE_DRIVE=str(cache), PASTA_RESULTADOS=str(raiz / "resultado1"),
                CAMINHO_LISTA_SUSPEITAS=str(raiz / "suspeitas.csv"),
                PERSISTIR_CACHES_NOVOS=False)
            with patch.multiple(c, **opcoes):
                tabela = preparar_cache(dados)
                self.assertEqual(len(tabela), 2)
                self.assertFalse(destino_zip.exists())
                self.assertEqual(legado.read_bytes(), legado_original)
                for resultado in ("resultado1", "resultado2"):
                    with patch.object(c, "PASTA_RESULTADOS", str(raiz / resultado)):
                        preparar_cache(dados)
                    self.assertTrue((raiz / resultado / "hashes_cache_treino.csv").is_file())
                    manifesto = json.loads((raiz / resultado / "manifesto_cache_treino.json").read_text())
                    self.assertEqual(manifesto["zip_disponivel"], str(legado))
                # Outra receita precisa processar as fotos, mas não deve criar
                # arquivos grandes no Drive na política padrão da campanha.
                with patch.multiple(c, QUALIDADE_JPEG=94,
                                    PASTA_IMAGENS_OTIMIZADAS=str(raiz / "receita_local"),
                                    PASTA_RESULTADOS=str(raiz / "resultado3")):
                    preparar_cache(dados)
                    self.assertFalse(destino_zip.exists())
                    self.assertTrue((raiz / "resultado3/hashes_cache_treino.csv").is_file())
                # A opção explícita continua disponível quando houver espaço.
                with patch.multiple(c, QUALIDADE_JPEG=94, PERSISTIR_CACHES_NOVOS=True,
                                    PASTA_IMAGENS_OTIMIZADAS=str(raiz / "receita_persistida"),
                                    PASTA_RESULTADOS=str(raiz / "resultado4")):
                    preparar_cache(dados)
                    self.assertTrue(destino_zip.is_file())
                    with zipfile.ZipFile(destino_zip) as arquivo:
                        self.assertIsNone(arquivo.testzip())
                self.assertEqual(legado.read_bytes(), legado_original)
            print("CACHE SINTÉTICO VERIFICADO: legado preservado; receitas locais; "
                  "hashes por execução; persistência opcional.")


if __name__ == "__main__":
    unittest.main()
