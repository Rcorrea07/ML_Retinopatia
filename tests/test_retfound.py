"""
Testes do RETFound-Green (`retfound`) que não precisam de GPU nem dos
pesos: o cosseno da taxa, as camadas da taxa por camada, os pesos das
classes, o aumento de dados e, com PyTorch e timm, a rede com pesos
aleatórios na CPU. Precisam de numpy, pandas e OpenCV (e de torch e timm
para a rede): pulam nesta máquina e rodam no Colab (notebook
Analisar_Resultados, que tem tudo isso instalado).

    python3 -m unittest discover -s tests -v
"""

import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _tem(*nomes):
    return all(importlib.util.find_spec(nome) is not None for nome in nomes)


TEM_BASICO = _tem("numpy", "pandas")
TEM_OPENCV = TEM_BASICO and _tem("cv2")
TEM_TORCH = TEM_BASICO and _tem("torch", "timm")


@unittest.skipUnless(TEM_BASICO, "numpy e pandas ausentes")
class RegrasDoTreino(unittest.TestCase):

    def setUp(self):
        from retinopatia import retfound
        self.r = retfound

    def test_cosseno_aquece_e_cai_ate_a_fracao_final(self):
        fator = self.r.fator_cosseno

        self.assertAlmostEqual(fator(0, 100, 10, 0.01), 0.1)
        self.assertAlmostEqual(fator(9, 100, 10, 0.01), 1.0)
        self.assertAlmostEqual(fator(10, 100, 10, 0.01), 1.0)
        self.assertAlmostEqual(fator(100, 100, 10, 0.01), 0.01)

        meio = fator(55, 100, 10, 0.01)
        self.assertAlmostEqual(meio, 0.01 + 0.99 * 0.5)

        valores = [fator(p, 100, 10, 0.01) for p in range(10, 101)]
        self.assertEqual(valores, sorted(valores, reverse=True))

    def test_camadas_da_taxa_por_camada(self):
        camada = self.r.camada_do_parametro

        self.assertEqual(camada("extrator.pos_embed", 12), 0)
        self.assertEqual(camada("extrator.reg_token", 12), 0)
        self.assertEqual(camada("extrator.patch_embed.proj.weight", 12), 0)
        self.assertEqual(camada("extrator.blocks.0.attn.qkv.weight", 12), 1)
        self.assertEqual(camada("extrator.blocks.11.mlp.fc2.bias", 12), 12)
        self.assertEqual(camada("extrator.norm.weight", 12), 13)
        self.assertEqual(camada("doente.weight", 12), 13)

    def test_pesos_das_classes_balanceados(self):
        pesos = self.r.pesos_classes([0, 0, 0, 1])

        self.assertAlmostEqual(pesos[0], 4 / 6)
        self.assertAlmostEqual(pesos[1], 2.0)

        with self.assertRaises(ValueError):
            self.r.pesos_classes([0, 0])


@unittest.skipUnless(TEM_OPENCV, "OpenCV ausente")
class AumentoDeDados(unittest.TestCase):

    def test_aumento_mantem_formato_faixa_e_fundo(self):
        import numpy as np

        from retinopatia import retfound

        img = np.full((392, 392, 3), 128, dtype=np.uint8)
        img[150:250, 150:250] = 220

        gerador = np.random.default_rng(0)

        for _ in range(20):
            saida = retfound.aumentar(
                img,
                gerador,
                {"fator_rotacao": 0.5, "fator_zoom": 0.15, "fator_brilho": 0.0, "fator_contraste": 0.0},
                128
            )

            self.assertEqual(saida.shape, img.shape)
            self.assertEqual(saida.dtype, np.float32)
            self.assertTrue(0 <= saida.min() and saida.max() <= 255)

            # O canto continua fundo cinza (rotação e zoom preenchem com 128)
            self.assertAlmostEqual(float(saida[0, 0, 0]), 128.0)


@unittest.skipUnless(TEM_TORCH, "torch e timm ausentes")
class RedeNaCpu(unittest.TestCase):

    def test_rede_e_grupos_da_taxa(self):
        import torch

        from retinopatia import configuracao, retfound

        rede = retfound.construir_rede(None).eval()

        imagens = torch.full((2, 3) + tuple(configuracao.TAMANHO_ENTRADA_RETFOUND), 128.0)

        with torch.inference_mode():
            logits, graus = rede(imagens)

        self.assertEqual(tuple(logits.shape), (2,))
        self.assertEqual(tuple(graus.shape), (2,))
        self.assertEqual(rede.extrator.num_features, 384)

        grupos = retfound.grupos_parametros(rede.named_parameters(), 12, 1e-4, 0.75, 0.05)

        nos_grupos = sum(len(g["params"]) for g in grupos)
        self.assertEqual(nos_grupos, len(list(rede.parameters())))

        taxas = {round(g["taxa_maxima"], 12) for g in grupos}
        self.assertAlmostEqual(max(taxas), 1e-4)
        self.assertAlmostEqual(min(taxas), 1e-4 * 0.75 ** 13)

        for grupo in grupos:
            if any(p.ndim == 1 for p in grupo["params"]):
                self.assertEqual(grupo["weight_decay"], 0.0)


if __name__ == "__main__":
    unittest.main()
