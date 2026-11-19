"""
Validação externa: o modelo escolhido medido uma única vez em bases que
nunca entraram no treino (o Messidor-2; depois, o mBRSET), com os
limiares que a validação do EyePACS escolheu.

Nada é escolhido aqui. Os limiares vêm das predições de validação que o
R09 gravou (`PASTA_RESULTADOS_R09`), com travas: o R09 avaliou o modelo
de `CAMINHO_MODELO_ESCOLHIDO`, no mesmo modo de TTA, e os limiares
recalculados batem com os do resumo dele. A GPU só prevê as fotos da
base externa; o resto é a avaliação de sempre (`avaliacao`) e o
diagnóstico (`diagnostico`).

Só as fotos graduáveis são medidas. Desfecho principal, fixado antes de
rodar (experimento `V01_r12_messidor2`): por paciente, só os pacientes
com os dois olhos graduáveis, qualquer retinopatia (graus 1 a 4),
sensibilidade e especificidade nos limiares do EyePACS e ROC-AUC, com
intervalos de 95% por bootstrap de pacientes. Os demais números (por
olho, todos os pacientes com algum olho graduável, retinopatia
referenciável, por grau e os três modos de TTA) são secundários.

Os resultados vão para a pasta da execução, com um `resumo_externo.csv`
(não `resumo.csv`: a validação externa não entra no registro de
experimentos de treino).
"""

import os

import pandas as pd

from . import ambiente_colab, avaliacao, configuracao, diagnostico, fontes


def limiares_do_modelo_final(pasta=None):
    """
    Limiares da validação do EyePACS para o modelo escolhido, refeitos a
    partir das predições que o R09 gravou em `pasta` (padrão:
    `PASTA_RESULTADOS_R09`). Para a execução se o R09 avaliou outro
    modelo ou outro modo de TTA, ou se os limiares refeitos não batem com
    os do resumo do R09.

    Retorna:
        (limiares no formato de `avaliacao.medir_predicoes`, tabela de
        predições da validação)
    """

    if pasta is None:
        pasta = configuracao.PASTA_RESULTADOS_R09

    resumo = pd.read_csv(os.path.join(pasta, "resumo.csv")).iloc[0]

    modelo_esperado = os.path.basename(configuracao.CAMINHO_MODELO_ESCOLHIDO)

    if resumo.get("modelo_avaliado") != modelo_esperado:
        raise RuntimeError(
            f"O R09 em {pasta} avaliou {resumo.get('modelo_avaliado')}, "
            f"não o modelo escolhido ({modelo_esperado})."
        )

    if resumo.get("modo_tta") != configuracao.MODO_TTA:
        raise RuntimeError(
            f"O R09 usou {resumo.get('modo_tta')}; este experimento usa "
            f"{configuracao.MODO_TTA}."
        )

    tabela = pd.read_csv(
        os.path.join(pasta, "predicoes_validacao_modelo_final.csv"),
        dtype={"image": str, "patient_id": str}
    )

    resultado = avaliacao.medir_predicoes(
        tabela,
        avaliacao.variantes_da_tabela(tabela),
        configuracao.MODO_TTA
    )

    for nivel in ("olho", "paciente"):
        refeito = resultado[f"limiar_{nivel}"]
        gravado = float(resumo[f"limiar_{nivel}"])

        if abs(refeito - gravado) > configuracao.TOLERANCIA_LIMIARES_R09:
            raise RuntimeError(
                f"Limiar por {nivel} refeito ({refeito:.6f}) diferente do "
                f"gravado no R09 ({gravado:.6f})."
            )

    print(
        f"Limiares da validação do EyePACS ({configuracao.MODO_TTA}): "
        f"olho {resultado['limiar_olho']:.4f}, "
        f"paciente {resultado['limiar_paciente']:.4f} "
        f"(conferidos com o R09 em {os.path.basename(pasta)})"
    )

    return avaliacao.limiares_da_validacao(resultado), tabela


def graduaveis(tabela):
    """
    Só as fotos graduáveis de uma base de validação externa, com o grau
    e a classe inteiros, como nas tabelas do EyePACS.
    """

    avaliadas = tabela[tabela["gradavel"]].reset_index(drop=True)

    avaliadas["level_original"] = avaliadas["level_original"].astype(int)
    avaliadas["target"] = avaliadas["target"].astype(int)

    return avaliadas


def prever_tabela(modelo, tabela):
    """
    Predições do modelo nas imagens processadas de `tabela` (coluna
    `pasta_processada`), nos três modos de TTA.
    """

    from . import pipeline_dados

    dados = pipeline_dados.criar_dataset(tabela, treino=False)

    return avaliacao.prever_variantes(modelo, dados)


def _linha_resumo(nome, resultado, principal, bootstrap, avaliadas, nao_graduaveis):
    """
    Uma linha com o que a execução mediu, para `resumo_externo.csv`.
    """

    linha = {
        "data": pd.Timestamp.now().isoformat(timespec="seconds"),
        "experimento": configuracao.NOME_EXPERIMENTO,
        "commit": configuracao.COMMIT_CODIGO,
        "modelo_avaliado": os.path.basename(configuracao.CAMINHO_MODELO_ESCOLHIDO),
        "base": nome,
        "modo_tta": resultado["modo_tta"],
        "limiar_olho": resultado["limiar_olho"],
        "limiar_paciente": resultado["limiar_paciente"],
        "fotos_avaliadas": len(avaliadas),
        "fotos_nao_graduaveis": nao_graduaveis,
        "pacientes": avaliadas["patient_id"].nunique(),
        "pacientes_dois_olhos": int((avaliadas.groupby("patient_id").size() == 2).sum())
    }

    for prefixo, medido in (("", resultado), ("principal_", principal)):
        for nivel in ("olho", "paciente"):
            for metrica in ("roc_auc", "pr_auc", "sensibilidade", "especificidade"):
                linha[f"{prefixo}{nivel}_{metrica}"] = round(
                    float(medido[f"metricas_{nivel}"][metrica]),
                    4
                )

    for metrica in ("roc_auc", "sensibilidade", "especificidade"):
        for limite in ("ic95_inferior", "ic95_superior"):
            linha[f"principal_paciente_{metrica}_{limite}"] = round(
                float(bootstrap.loc[("paciente", metrica), limite]),
                4
            )

    for modo, medidas in principal["comparacao_tta"].iterrows():
        for metrica in ("roc_auc", "sensibilidade", "especificidade"):
            linha[f"principal_paciente_{metrica}_{modo}"] = round(float(medidas[metrica]), 4)

    return linha


def medir_base(
    nome,
    avaliadas,
    variantes,
    limiares,
    tabela_validacao,
    nao_graduaveis=0,
    pasta=None
):
    """
    Tudo o que a validação externa mede numa base, a partir das
    predições (`variantes`, alinhadas com `avaliadas`): a avaliação
    completa (todos os pacientes com algum olho graduável), o desfecho
    principal (pacientes com os dois olhos graduáveis) com intervalos por
    bootstrap, os erros por grau e a retinopatia referenciável (limiares
    desse alvo escolhidos na validação do EyePACS). Grava as tabelas e o
    `resumo_externo.csv` em `pasta` (padrão: a pasta da execução).

    Retorna um dicionário com as partes.
    """

    if pasta is None:
        pasta = configuracao.PASTA_RESULTADOS

    resultado = avaliacao.avaliar_predicoes(
        avaliadas,
        variantes,
        nome.lower(),
        nome,
        "validação externa",
        modo_tta=configuracao.MODO_TTA,
        limiares=limiares,
        pasta_resultados=pasta
    )

    dois_olhos = (
        avaliadas.groupby("patient_id")["image"].transform("size") == 2
    ).values

    principal = avaliacao.medir_predicoes(
        avaliadas[dois_olhos].reset_index(drop=True),
        {
            modo: (probabilidades[dois_olhos], graus[dois_olhos])
            for modo, (probabilidades, graus) in variantes.items()
        },
        configuracao.MODO_TTA,
        limiares=limiares
    )

    print(
        f"\nDesfecho principal ({nome}, {dois_olhos.sum() // 2} pacientes com os "
        "dois olhos graduáveis, por paciente): "
        f"ROC-AUC {principal['metricas_paciente']['roc_auc']:.4f}, "
        f"sensibilidade {principal['metricas_paciente']['sensibilidade']:.4f}, "
        f"especificidade {principal['metricas_paciente']['especificidade']:.4f}"
    )

    tabela = resultado["tabela"]

    partes = {
        "por_grau": diagnostico.relatorio_por_grau(
            tabela,
            limiares["olho"],
            nome
        ),
        "referenciavel": diagnostico.avaliar_referenciavel(
            tabela_validacao,
            tabela,
            nome_medido=nome
        ),
        "bootstrap_principal": diagnostico.intervalo_bootstrap(
            tabela[dois_olhos].reset_index(drop=True),
            limiares["olho"],
            limiares["paciente"],
            f"{nome}, pacientes com os dois olhos graduáveis"
        ),
        "modos_tta_principal": principal["comparacao_tta"]
    }

    os.makedirs(pasta, exist_ok=True)

    for parte, tabela_parte in partes.items():
        tabela_parte.to_csv(
            os.path.join(pasta, f"validacao_externa_{nome.lower()}_{parte}.csv")
        )

    resumo = pd.DataFrame([
        _linha_resumo(
            nome,
            resultado,
            principal,
            partes["bootstrap_principal"],
            avaliadas,
            nao_graduaveis
        )
    ])

    caminho = os.path.join(pasta, "resumo_externo.csv")
    resumo.to_csv(caminho, index=False)

    print(f"\nResumo da validação externa salvo em: {caminho}")

    return {"avaliacao": resultado, "principal": principal, "resumo": resumo, **partes}


def rodar_messidor2(desligar_ao_terminar=True):
    """
    A validação externa no Messidor-2 numa sessão: limiares do R09,
    imagens do Messidor-2 (extraídas e processadas nesta sessão, se não
    houver cache), predição do modelo escolhido e a medida. Com
    `desligar_ao_terminar`, encerra a sessão no fim, também se algo der
    errado.

    Retorna o resultado de `medir_base`.
    """

    from . import utilitarios
    from .modelo import carregar_modelo_salvo

    try:
        os.makedirs(configuracao.PASTA_RESULTADOS, exist_ok=True)

        limiares, tabela_validacao = limiares_do_modelo_final()

        tabela = fontes.preparar_fonte(fontes.fonte_validacao_externa("Messidor2"))

        avaliadas = graduaveis(tabela)

        print(
            f"Fotos medidas: {len(avaliadas)} graduáveis; "
            f"{len(tabela) - len(avaliadas)} não graduáveis ficam de fora."
        )

        print(f"Modelo avaliado: {configuracao.CAMINHO_MODELO_ESCOLHIDO}")

        modelo = carregar_modelo_salvo(configuracao.CAMINHO_MODELO_ESCOLHIDO)

        if tuple(modelo.input_shape[1:]) != tuple(configuracao.FORMATO_ENTRADA):
            raise ValueError(
                f"O modelo espera {modelo.input_shape[1:]}, mas o experimento "
                f"prepara imagens {configuracao.FORMATO_ENTRADA}."
            )

        utilitarios.salvar_configuracao()

        return medir_base(
            "Messidor2",
            avaliadas,
            prever_tabela(modelo, avaliadas),
            limiares,
            tabela_validacao,
            nao_graduaveis=len(tabela) - len(avaliadas)
        )

    finally:
        if desligar_ao_terminar:
            ambiente_colab.desligar_ambiente()
