"""Lê os artefatos da campanha sem importar ML, prever ou alterar um treino.

Também pode ser executado como script avulso em um terminal do Colab.
"""

import argparse
import csv
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


RAIZ_PADRAO = "/content/drive/MyDrive/ProjetoRetinopatia"
AVALIACOES = {"fase1_sem_tta": "metricas_validacao_fase1.json",
              "fase2_sem_tta": "metricas_validacao_fase2.json",
              "fase2_tta4": "metricas_validacao_tta4.json"}
COLUNAS = ["perfil", "experimento", "estado", "avaliacao", "nivel", "provisorio",
           "roc_auc", "pr_auc_ap", "sensibilidade", "especificidade", "limiar"]


def _json(caminho):
    return json.loads(caminho.read_text(encoding="utf-8"))


def ler_campanha(raiz=RAIZ_PADRAO):
    """Arquivos ausentes durante o treino significam pendência, não nota zero."""
    resultados = Path(raiz) / "resultados"
    if not resultados.is_dir():
        raise FileNotFoundError(f"Pasta de resultados não encontrada: {resultados}")
    estados, metricas, avisos = [], [], []
    for pasta in sorted(resultados.iterdir()):
        if not pasta.is_dir() or not pasta.name.startswith("E"):
            continue
        try:
            config = _json(pasta / "configuracao.json")
            perfil = config.get("PERFIL_EXPERIMENTO", "")
            if not perfil.startswith("E") or config.get("ENSAIO", False):
                continue
            estado = (_json(pasta / "estado_execucao.json")
                      if (pasta / "estado_execucao.json").exists() else {})
            item = {"perfil": perfil, "experimento": pasta.name,
                    "estado": estado.get("estado", "sem_estado"),
                    "atualizado_em": estado.get("atualizado_em"), "pasta": str(pasta),
                    "commit": config.get("COMMIT_CODIGO"), "fases": {}}
            for fase in (1, 2):
                arquivo = pasta / f"estado_fase{fase}.json"
                etapa = _json(arquivo) if arquivo.exists() else {}
                item["fases"][str(fase)] = {
                    "epocas_concluidas": etapa.get("epocas_concluidas", 0),
                    "concluida": etapa.get("concluida", False),
                    "snapshot_existe": (pasta / f"retomada_fase{fase}.keras").is_file(),
                    "melhor_peso_existe": (pasta / f"melhor_modelo_fase{fase}.weights.h5").is_file()}
            estados.append(item)
            for avaliacao, nome in AVALIACOES.items():
                arquivo = pasta / nome
                if not arquivo.exists():
                    continue
                dados = _json(arquivo)
                for nivel in ("olho", "paciente"):
                    m = dados[f"metricas_{nivel}"]
                    metricas.append({"perfil": perfil, "experimento": pasta.name,
                        "estado": item["estado"], "avaliacao": avaliacao, "nivel": nivel,
                        "provisorio": item["estado"] != "concluido" or avaliacao.startswith("fase1"),
                        "roc_auc": m["roc_auc"], "pr_auc_ap": m["pr_auc"],
                        "sensibilidade": m["sensibilidade"],
                        "especificidade": m["especificidade"], "limiar": m["limiar"]})
        except (OSError, ValueError, KeyError, TypeError) as erro:
            avisos.append(f"{pasta.name}: leitura incompleta ({type(erro).__name__}).")
    return {"estados": estados, "metricas": metricas, "avisos": avisos}


def _celula(valor):
    return str(valor).replace("|", "\\|").replace("\n", " ")


def gerar_relatorio(raiz=RAIZ_PADRAO, destino=None):
    """Gera um retrato novo em MD/CSV/JSON, preservando todos os originais."""
    dados = ler_campanha(raiz)
    instante = datetime.now(timezone.utc)
    pasta = Path(destino or Path(raiz) / "relatorios/campanha_2026-10-20") / (
        instante.strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8])
    pasta.mkdir(parents=True, exist_ok=False)
    (pasta / "estado_campanha.json").write_text(
        json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")
    with (pasta / "comparativo_validacao.csv").open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=COLUNAS)
        escritor.writeheader()
        escritor.writerows(dados["metricas"])
    texto = ["# Acompanhamento da campanha de retinopatia", "",
        f"Retrato gerado em {instante.isoformat()} (UTC). Atualize para ler os arquivos novos.", "",
        "Este relatório lê arquivos salvos; não confirma atividade da GPU em tempo real.",
        "Fase 1 = ajuste das cabeças; fase 2 = ajuste da rede. Ausência de métricas não é zero.", "",
        "| Perfil | Estado salvo | Épocas fase 1 / 2 | Snapshot fase 1 / 2 |",
        "| --- | --- | --- | --- |"]
    for item in dados["estados"]:
        f1, f2 = item["fases"]["1"], item["fases"]["2"]
        texto.append(f"| {_celula(item['perfil'])} | {_celula(item['estado'])} | "
                     f"{f1['epocas_concluidas']} / {f2['epocas_concluidas']} | "
                     f"{f1['snapshot_existe']} / {f2['snapshot_existe']} |")
    for avaliacao in AVALIACOES:
        texto += ["", f"## Validação: {avaliacao}", "",
                  "| Perfil | Nível | ROC-AUC | AP | Sensibilidade | Especificidade | Provisório |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        linhas = [m for m in dados["metricas"] if m["avaliacao"] == avaliacao]
        for m in linhas:
            texto.append(f"| {_celula(m['perfil'])} | {m['nivel']} | {m['roc_auc']:.4f} | "
                f"{m['pr_auc_ap']:.4f} | {m['sensibilidade']:.4f} | "
                f"{m['especificidade']:.4f} | {m['provisorio']} |")
        if not linhas:
            texto.append("| Ainda sem avaliação salva | | | | | | |")
    texto += ["", "## Como interpretar", "",
        "Compare fase 2 sem TTA entre os modelos concluídos; compare TTA4 em tabela separada.",
        "AP é average precision da avaliação; a PR-AUC aproximada do Keras no histórico usa outro cálculo.",
        "A sensibilidade/especificidade do relatório usam limiar escolhido na validação para sensibilidade ≥90%.",
        "Accuracy/precision/recall no histórico de treino usam o limiar padrão 0,5.",
        "Priorize AP/ROC-AUC e especificidade na sensibilidade desejada, verificando erros por grau.",
        "O checkpoint é escolhido pela PR-AUC de validação. A seleção repetida na validação pode ser otimista.",
        "O teste histórico não deve orientar a seleção; ganho independente exige outra amostra retida.", "",
        "## Arquivos para análise detalhada", "",
        "Em cada pasta abaixo ficam os históricos por época, figuras PNG, predições por olho, "
        "matrizes, limiares, configuração e checkpoints. Ao concluir, também haverá bootstrap por paciente, "
        "diagnóstico por grau/cabeças/agregadores e modelo final. E11 acrescenta cinco classes 0–4 e QWK.", ""]
    texto += [f"- `{_celula(e['pasta'])}`" for e in dados["estados"]]
    texto += ["", "## Avisos de leitura", "", *(dados["avisos"] or ["Nenhum."])]
    relatorio = pasta / "LEIA_RESULTADOS.md"
    relatorio.write_text("\n".join(texto) + "\n", encoding="utf-8")
    print(f"Relatório: {relatorio}")
    for item in dados["estados"]:
        print(item["perfil"], item["estado"], "épocas", {
            f: etapa["epocas_concluidas"] for f, etapa in item["fases"].items()})
    return relatorio


def mostrar_figuras(pasta_experimento, limite=20):
    """Mostra no Colab os gráficos já salvos, sem gerar novas predições."""
    from IPython.display import Image, display
    for imagem in sorted(Path(pasta_experimento).glob("figura_*.png"))[:limite]:
        print(imagem.name)
        display(Image(filename=str(imagem)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raiz", default=RAIZ_PADRAO)
    parser.add_argument("--destino")
    args = parser.parse_args()
    gerar_relatorio(args.raiz, args.destino)
