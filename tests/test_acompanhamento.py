"""O relatório lê artefatos; não importa TensorFlow nem executa ML."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from retinopatia.acompanhamento import gerar_relatorio, ler_campanha


class Acompanhamento(unittest.TestCase):
    def montar(self, raiz, nome="E01_treino", ensaio=False):
        pasta = raiz / "resultados" / nome
        pasta.mkdir(parents=True)
        (pasta / "configuracao.json").write_text(json.dumps({
            "PERFIL_EXPERIMENTO": "E01", "ENSAIO": ensaio, "COMMIT_CODIGO": "codigo"}))
        (pasta / "estado_execucao.json").write_text('{"estado":"treinando_fase2"}')
        return pasta

    def test_separa_avaliacoes_e_preserva_arquivos(self):
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            pasta = self.montar(raiz)
            metricas = dict(roc_auc=.8, pr_auc=.7, sensibilidade=.9, especificidade=.5, limiar=.3)
            for arquivo in ("metricas_validacao_fase1.json", "metricas_validacao_tta4.json"):
                (pasta / arquivo).write_text(json.dumps({
                    "metricas_olho": metricas, "metricas_paciente": metricas}))
            originais = {p.name: p.read_bytes() for p in pasta.iterdir()}
            relatorio = gerar_relatorio(raiz)
            segundo = gerar_relatorio(raiz)
            self.assertNotEqual(relatorio, segundo)
            self.assertEqual(originais, {p.name: p.read_bytes() for p in pasta.iterdir()})
            with (relatorio.parent / "comparativo_validacao.csv").open() as arquivo:
                linhas = list(csv.DictReader(arquivo))
            self.assertEqual(len(linhas), 4)
            self.assertEqual({l["avaliacao"] for l in linhas}, {"fase1_sem_tta", "fase2_tta4"})
            self.assertTrue(all(l["provisorio"] == "True" for l in linhas))

    def test_ensaio_excluido_e_metrica_ausente_nao_vira_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            self.montar(raiz)
            self.montar(raiz, "E01_ensaio", ensaio=True)
            dados = ler_campanha(raiz)
            self.assertEqual(len(dados["estados"]), 1)
            self.assertEqual(dados["metricas"], [])
            self.assertFalse(dados["estados"][0]["fases"]["1"]["snapshot_existe"])

    def test_leitura_interrompida_fica_explicita(self):
        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            pasta = self.montar(raiz)
            (pasta / "metricas_validacao_fase2.json").write_text('{"metricas')
            dados = ler_campanha(raiz)
            self.assertEqual(len(dados["avisos"]), 1)
            self.assertEqual(dados["metricas"], [])


if __name__ == "__main__":
    unittest.main()
