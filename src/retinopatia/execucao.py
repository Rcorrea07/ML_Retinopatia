"""
O pipeline de um treino e vários treinos numa sessão só do Colab.

As etapas de um treino existem uma vez, aqui, e servem ao notebook de
treino (célula a célula) e a cada processo dos treinos paralelos
(`executar_experimento`):

1. `iniciar_execucao()`: sementes e precisão mista;
2. `preparar_dados()`: imagens do EyePACS (cache ou extração e
   processamento), divisão por paciente, fotos só de treino (extras de
   2015 e datasets externos) e os ajustes do experimento (fumaça, pares,
   destilação);
3. `treinar()`: pipelines, pesos das classes, modelo, Fase 1, Fase 2 e
   o modelo final salvo;
4. `avaliar_e_registrar()`: validação, teste (só no modelo final),
   resumo da execução e diagnóstico.

Uma sessão A100 do Colab cobra por hora (~6,8 unidades), use a GPU toda
ou não. Um treino em 768 px ocupa ~13 dos 80 GB de VRAM e ~25 dos 167 GB
de RAM. Por isso há também os treinos paralelos:

1. `preparar_imagens()` extrai e processa as imagens uma vez, no tamanho
   do experimento escolhido na primeira célula do notebook;
2. `treinar_em_paralelo()` roda cada treino (experimento + semente) num
   processo próprio (`python -m retinopatia.execucao`), quantos couberem
   na memória ao mesmo tempo; os outros esperam na fila. Cada processo
   escolhe o experimento pelas variáveis de ambiente antes do import,
   como o notebook de treino, e aloca a memória da GPU sob demanda;
3. `rodar_rodada()` junta as etapas, confere antes cada treino com uma
   execução de fumaça, consolida o registro e desliga a sessão no fim,
   também se algo der errado.

Todos os treinos de uma sessão usam o mesmo tamanho de imagem. Este
módulo não importa TensorFlow ao ser importado: o processo do notebook
dos treinos paralelos não o carrega, e a GPU fica para os treinos.
"""

import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

from . import ambiente_colab, configuracao

# Pasta `src` do repositório, posta no PYTHONPATH dos processos de treino
PASTA_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Imagens do EyePACS que `preparar_imagens_eyepacs` processa
CONJUNTOS_A_PROCESSAR = ("todos", "avaliacao", "validacao", None)


# ============================================================
# O PIPELINE DE UM TREINO
# ============================================================


@dataclass
class ConjuntosExperimento:
    """
    As tabelas de um experimento, prontas para os pipelines: `dados`
    (todas as fotos rotuladas do train do EyePACS), os três conjuntos já
    ajustados ao experimento e o status do pré-processamento de cada
    imagem (None quando as imagens já estavam no disco).
    """

    dados: object
    df_treino: object
    df_val: object
    df_teste: object
    status_imagens: object = None


@dataclass
class TreinoConcluido:
    """
    O que o treino deixa para a avaliação: o modelo, os pipelines de
    validação e teste, os históricos das duas fases e, se pedida, a
    avaliação da Fase 1.
    """

    modelo: object
    val_data: object
    teste_data: object
    historico_fase1: object
    historico_fase2: object
    resultado_fase1: object = None


def iniciar_execucao():
    """
    Sementes do treino e precisão mista. Vem antes de qualquer modelo.
    """

    from . import utilitarios

    print("Experimento:", configuracao.NOME_EXPERIMENTO)

    utilitarios.configurar_sementes(configuracao.SEED_TREINO)

    utilitarios.configurar_precisao_mista()


def preparar_imagens_eyepacs(processar="todos", caminho_lista_suspeitas=None):
    """
    Imagens do EyePACS no disco: restaura o cache do Drive (e extrai só
    os rótulos) ou extrai as fotos, monta as tabelas e a divisão por
    paciente e processa as imagens de `processar`: "todos" (o train
    inteiro), "avaliacao" (validação e teste), "validacao" ou None.
    A lista de máscaras suspeitas vai para `caminho_lista_suspeitas`
    (padrão: a da configuração).

    Retorna:
        (dados, df_treino, df_val, df_teste, status das imagens
        processadas ou None)
    """

    import pandas as pd

    from . import conjunto_dados, pre_processamento

    if processar not in CONJUNTOS_A_PROCESSAR:
        raise ValueError(
            f"processar deve ser um de {CONJUNTOS_A_PROCESSAR}, não {processar!r}"
        )

    cache_restaurado = ambiente_colab.restaurar_cache_imagens()

    ambiente_colab.extrair_dataset(somente_rotulos=cache_restaurado)

    dados, df_treino, df_val, df_teste = conjunto_dados.preparar_dataset()

    tabelas = {
        "todos": [dados],
        "avaliacao": [df_val, df_teste],
        "validacao": [df_val],
        None: []
    }[processar]

    status = None

    if tabelas:
        opcoes = (
            {}
            if caminho_lista_suspeitas is None
            else {"caminho_lista_suspeitas": caminho_lista_suspeitas}
        )

        status = pre_processamento.pre_processar_dataset(
            pd.concat(tabelas) if len(tabelas) > 1 else tabelas[0],
            **opcoes
        )

    return dados, df_treino, df_val, df_teste, status


def _restaurar_imagens_preparadas():
    """
    Nos treinos paralelos, as imagens já foram processadas pelo notebook
    (`preparar_imagens`) e estão no cache do Drive: cada processo só as
    restaura (train, extras e datasets externos do experimento) e extrai
    os rótulos. Falta de cache aqui é erro, não motivo para processar.
    """

    pastas = [(configuracao.PASTA_IMAGENS_OTIMIZADAS, configuracao.CAMINHOS_ZIP_CACHE)]

    if configuracao.USAR_DADOS_EXTRA:
        pastas.append((
            configuracao.PASTA_IMAGENS_OTIMIZADAS_EXTRA,
            configuracao.CAMINHOS_ZIP_CACHE_EXTRA
        ))

    for nome in configuracao.DATASETS_EXTERNOS:
        pastas.append((
            configuracao.PASTAS_IMAGENS_EXTERNAS[nome],
            configuracao.CAMINHOS_ZIP_CACHE_EXTERNOS[nome]
        ))

    for pasta, caminhos_zip in pastas:
        if not ambiente_colab.restaurar_cache_imagens(pasta, caminhos_zip):
            raise RuntimeError(
                f"Faltam as imagens processadas em {pasta}: todos os treinos da "
                "sessão precisam do tamanho preparado por preparar_imagens()."
            )

    ambiente_colab.extrair_rotulos()


def preparar_dados(imagens_no_disco=False):
    """
    As tabelas do experimento, com as imagens no disco: as do EyePACS
    (`preparar_imagens_eyepacs`, ou só restauradas quando
    `imagens_no_disco`), as fotos que só entram no treino e os ajustes do
    experimento (`treinamento.preparar_conjuntos_experimento`, que na
    destilação usa a GPU).

    Retorna um `ConjuntosExperimento`.
    """

    from . import conjunto_dados, treinamento

    if imagens_no_disco:
        _restaurar_imagens_preparadas()

        dados, df_treino, df_val, df_teste = conjunto_dados.preparar_dataset()
        status = None

    else:
        dados, df_treino, df_val, df_teste, status = preparar_imagens_eyepacs()

        # Grava o cache no Drive só no notebook Preparar_Caches (conta pessoal)
        ambiente_colab.salvar_cache_imagens()

    # Fotos só para o treino: extras de 2015 e datasets externos do
    # experimento (cache do Drive se existir, senão processadas aqui)
    df_adicionais = ambiente_colab.preparar_dados_adicionais(dados)

    if df_adicionais is not None:
        df_treino = conjunto_dados.ampliar_treino(df_treino, df_adicionais)

    df_treino, df_val, df_teste = treinamento.preparar_conjuntos_experimento(
        df_treino,
        df_val,
        df_teste
    )

    return ConjuntosExperimento(dados, df_treino, df_val, df_teste, status)


def treinar(conjuntos, avaliar_fase1=False, medir_velocidade=False):
    """
    O treino de um experimento: pipelines (com a conferência da escala),
    pesos das classes, modelo, Fase 1 (só as cabeças), Fase 2
    (fine-tuning) e o modelo final salvo e conferido. A configuração da
    execução é gravada no Drive antes da Fase 1.

    `avaliar_fase1` mostra a avaliação da Fase 1 na validação (notebook);
    `medir_velocidade` mede quantos lotes por segundo o pipeline entrega.

    Retorna um `TreinoConcluido`.
    """

    from . import avaliacao, pipeline_dados, treinamento, utilitarios
    from .modelo import construir_modelo, salvar_modelo_final

    treino_data, val_data, teste_data = pipeline_dados.criar_geradores(
        conjuntos.df_treino,
        conjuntos.df_val,
        conjuntos.df_teste
    )

    pipeline_dados.verificar_escala(treino_data, val_data)

    # Se o pipeline levar perto de 1 s por lote, a GPU vai ficar esperando
    if medir_velocidade:
        pipeline_dados.medir_velocidade(treino_data)

    pesos_dicionario = treinamento.calcular_pesos_classe(conjuntos.df_treino)

    modelo, modelo_base = construir_modelo()

    # Registra no Drive os hiperparâmetros deste experimento
    utilitarios.salvar_configuracao()

    historico_fase1 = treinamento.treinar_fase1(
        modelo,
        treino_data,
        val_data,
        pesos_dicionario,
        treinamento.criar_callbacks_fase1()
    )

    resultado_fase1 = None

    if avaliar_fase1:
        resultado_fase1 = avaliacao.avaliar_modelo(
            modelo,
            val_data,
            conjuntos.df_val,
            historico_fase1,
            "Fase 1"
        )

    historico_fase2 = treinamento.fine_tuning_fase2(
        modelo,
        modelo_base,
        treino_data,
        val_data,
        pesos_dicionario,
        treinamento.criar_callbacks_fase2()
    )

    # Salva o modelo completo (arquitetura + pesos) e confere a gravação
    # recarregando o arquivo
    salvar_modelo_final(modelo, val_data)

    return TreinoConcluido(
        modelo,
        val_data,
        teste_data,
        historico_fase1,
        historico_fase2,
        resultado_fase1
    )


def avaliar_e_registrar(treino, conjuntos):
    """
    Avaliação da Fase 2 na validação (limiares), o teste (só com
    RETINOPATIA_AVALIAR_TESTE=1, modelo final), o resumo da execução e o
    diagnóstico completo.

    Retorna:
        {"validacao": ..., "teste": ... ou None, "diagnostico": ...}
    """

    from . import avaliacao, diagnostico

    resultado_fase2 = avaliacao.avaliar_modelo(
        treino.modelo,
        treino.val_data,
        conjuntos.df_val,
        treino.historico_fase2,
        "Fase 2"
    )

    resultado_teste = avaliacao.avaliar_no_teste(
        treino.modelo,
        treino.teste_data,
        conjuntos.df_teste,
        resultado_fase2
    )

    avaliacao.registrar_experimento(resultado_fase2, resultado_teste)

    return {
        "validacao": resultado_fase2,
        "teste": resultado_teste,
        "diagnostico": diagnostico.relatorio_completo(
            resultado_fase2,
            resultado_teste
        )
    }


# ============================================================
# TREINOS PARALELOS: NO PROCESSO DO NOTEBOOK
# ============================================================

def preparar_imagens():
    """
    Deixa no disco local as imagens processadas do tamanho do experimento
    atual (train e, se o experimento os usa, o test de 2015 e os datasets
    externos), com as mesmas etapas de `preparar_dados`: restaura o
    cache se houver, senão extrai e processa. Não monta os conjuntos do
    experimento (a destilação usaria a GPU neste processo).
    """

    dados, *_ = preparar_imagens_eyepacs()

    ambiente_colab.preparar_dados_adicionais(dados)


def _memoria_total_gb():
    """
    RAM total da máquina, em GB.
    """

    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9


def _estado_gpu():
    """
    (uso da GPU em %, VRAM usada em GB, VRAM total em GB), pelo
    nvidia-smi, ou None quando não há GPU.
    """

    try:
        linha = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            check=True
        ).stdout.splitlines()[0]

    except (OSError, subprocess.CalledProcessError, IndexError):
        return None

    uso, usada, total = (float(valor) for valor in linha.split(","))

    return uso, usada / 1024, total / 1024


def simultaneos_possiveis(quantidade, ram_gb=None, vram_gb=None):
    """
    Quantos treinos cabem ao mesmo tempo na RAM e na VRAM da máquina,
    pela memória reservada por treino em `configuracao` (pelo menos 1,
    no máximo `quantidade`).
    """

    if ram_gb is None:
        ram_gb = _memoria_total_gb()

    if vram_gb is None:
        estado = _estado_gpu()
        vram_gb = estado[2] if estado else configuracao.VRAM_POR_TREINO_GB

    por_ram = int(
        (ram_gb - configuracao.FOLGA_RAM_GB) // configuracao.RAM_POR_TREINO_GB
    )
    por_vram = int(vram_gb // configuracao.VRAM_POR_TREINO_GB)

    return max(1, min(quantidade, por_ram, por_vram))


def ambiente_do_treino(experimento, semente, fumaca):
    """
    Variáveis de ambiente de um processo de treino. O ID da execução vem
    do notebook, então todos os treinos da sessão o compartilham (as
    pastas se distinguem pelo experimento e pela semente).
    """

    ambiente = dict(os.environ)

    ambiente.update({
        "RETINOPATIA_EXPERIMENTO": experimento,
        "RETINOPATIA_SEED_TREINO": str(semente),
        "RETINOPATIA_FUMACA": "1" if fumaca else "0",
        "RETINOPATIA_AVALIAR_TESTE": "0",
        # Sem isso, o primeiro processo reservaria a VRAM toda
        "TF_FORCE_GPU_ALLOW_GROWTH": "true",
        # Sem tela: os gráficos são só gravados
        "MPLBACKEND": "Agg",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": os.pathsep.join(
            caminho
            for caminho in (PASTA_SRC, ambiente.get("PYTHONPATH"))
            if caminho
        ),
    })

    return ambiente


def _andamento(caminho_log):
    """
    Resumo do log de um treino: a última época iniciada e a fase.
    """

    try:
        with open(caminho_log, encoding="utf-8", errors="replace") as arquivo:
            linhas = [linha.strip() for linha in arquivo]

    except OSError:
        return "sem log ainda"

    epocas = [linha for linha in linhas if re.fullmatch(r"Epoch \d+/\d+", linha)]

    if epocas:
        fase = "Fase 2" if any("Fase 2" in linha for linha in linhas) else "Fase 1"
        return f"{fase}, época {epocas[-1].split()[1]}"

    ultima = next((linha for linha in reversed(linhas) if linha), "")

    return ultima[:100] or "iniciando"


def _mostrar_andamento(rodando):
    """
    Uma linha por treino em andamento, mais o uso da GPU.
    """

    estado = _estado_gpu()

    gpu = (
        f" | GPU: {estado[0]:.0f}% de uso, {estado[1]:.0f} de {estado[2]:.0f} GB"
        if estado else ""
    )

    print(f"\n[{time.strftime('%H:%M')}] Andamento{gpu}")

    for (experimento, semente), (_, _, caminho, inicio) in rodando.items():
        horas = (time.time() - inicio) / 3600
        print(f"  {experimento}, semente {semente} ({horas:.1f} h): {_andamento(caminho)}")


def _copiar_log(caminho, pasta_destino):
    """
    Copia o log de um treino para o Drive, sem interromper a rodada se
    a cópia falhar.
    """

    try:
        os.makedirs(pasta_destino, exist_ok=True)
        shutil.copy(caminho, pasta_destino)

    except OSError as erro:
        print(f"Não foi possível copiar {caminho} para o Drive: {erro}")


def treinar_em_paralelo(
    execucoes,
    simultaneos=None,
    fumaca=None,
    intervalo_minutos=None
):
    """
    Roda cada (experimento, semente) de `execucoes` num processo próprio,
    até `simultaneos` ao mesmo tempo na mesma GPU (pela memória da
    máquina, se None). Mostra o andamento a cada `intervalo_minutos`.

    Retorna:
        {(experimento, semente): código de saída}, 0 = concluído
    """

    if fumaca is None:
        fumaca = configuracao.FUMACA

    if intervalo_minutos is None:
        intervalo_minutos = configuracao.INTERVALO_ACOMPANHAMENTO_MIN

    execucoes = [tuple(execucao) for execucao in execucoes]

    if len(set(execucoes)) != len(execucoes):
        raise ValueError(f"Há treinos repetidos na lista: {execucoes}")

    if simultaneos is None:
        simultaneos = simultaneos_possiveis(len(execucoes))

    print(
        f"{len(execucoes)} treinos{' de fumaça' if fumaca else ''}, "
        f"até {simultaneos} ao mesmo tempo nesta GPU."
    )

    os.makedirs(configuracao.PASTA_LOGS_LOCAL, exist_ok=True)

    pasta_logs_drive = os.path.join(
        configuracao.PASTA_LOGS_PARALELOS,
        configuracao.ID_EXECUCAO
    )

    fila = list(execucoes)
    rodando = {}
    codigos = {}
    ultimo_relato = time.time()

    try:
        while fila or rodando:
            while fila and len(rodando) < simultaneos:
                experimento, semente = execucao = fila.pop(0)

                caminho = os.path.join(
                    configuracao.PASTA_LOGS_LOCAL,
                    f"{experimento}_seed{semente}{'_fumaca' if fumaca else ''}.log"
                )

                arquivo = open(caminho, "w", encoding="utf-8")

                processo = subprocess.Popen(
                    [sys.executable, "-m", "retinopatia.execucao"],
                    env=ambiente_do_treino(experimento, semente, fumaca),
                    stdout=arquivo,
                    stderr=subprocess.STDOUT
                )

                rodando[execucao] = (processo, arquivo, caminho, time.time())

                print(f"Iniciado: {experimento}, semente {semente} (log: {caminho})")

            for execucao, (processo, arquivo, caminho, inicio) in list(rodando.items()):
                codigo = processo.poll()

                if codigo is None:
                    continue

                arquivo.close()
                del rodando[execucao]
                codigos[execucao] = codigo

                _copiar_log(caminho, pasta_logs_drive)

                estado = "concluído" if codigo == 0 else f"ERRO (código {codigo}), veja o log"
                horas = (time.time() - inicio) / 3600

                print(f"{execucao[0]}, semente {execucao[1]}: {estado} em {horas:.1f} h")

            if rodando and time.time() - ultimo_relato >= intervalo_minutos * 60:
                _mostrar_andamento(rodando)
                ultimo_relato = time.time()

            if fila or rodando:
                time.sleep(30)

    except BaseException:
        # Interrompido (ou erro no notebook): nenhum treino fica órfão
        for processo, arquivo, caminho, _ in rodando.values():
            processo.terminate()
            arquivo.close()
            _copiar_log(caminho, pasta_logs_drive)

        raise

    return codigos


def rodar_rodada(
    execucoes,
    simultaneos=None,
    fumaca_antes=True,
    desligar_ao_terminar=True
):
    """
    Uma rodada inteira numa sessão: prepara as imagens uma vez, confere
    cada treino com uma execução de fumaça (minutos; evita perder a
    rodada por um erro bobo), roda os treinos de verdade, consolida o
    registro e, com `desligar_ao_terminar`, encerra a sessão no fim,
    também se algo der errado.

    Retorna:
        {(experimento, semente): código de saída}, 0 = concluído
    """

    try:
        preparar_imagens()

        if fumaca_antes and not configuracao.FUMACA:
            print("\nConferência rápida (fumaça) de cada treino:")

            codigos = treinar_em_paralelo(execucoes, simultaneos, fumaca=True)
            falhas = [execucao for execucao, codigo in codigos.items() if codigo]

            if falhas:
                raise RuntimeError(
                    f"A fumaça falhou em {falhas}; os logs estão em "
                    f"{configuracao.PASTA_LOGS_LOCAL}. A rodada não começou."
                )

            print("\nFumaça concluída. Começando os treinos:")

        codigos = treinar_em_paralelo(execucoes, simultaneos)

        from . import avaliacao

        avaliacao.consolidar_experimentos()

        falhas = [execucao for execucao, codigo in codigos.items() if codigo]

        if falhas:
            print(f"\nTreinos com erro: {falhas}. Logs em {configuracao.PASTA_LOGS_PARALELOS}.")

        return codigos

    finally:
        if desligar_ao_terminar:
            ambiente_colab.desligar_ambiente()


# ============================================================
# NO PROCESSO DE CADA TREINO
# ============================================================

def executar_experimento():
    """
    Um treino completo, do experimento e da semente das variáveis de
    ambiente: as etapas do pipeline, sem os gráficos de inspeção e a
    avaliação da Fase 1. Supõe o Drive montado e as imagens já
    processadas no cache (`preparar_imagens`).
    """

    if not os.path.isdir(configuracao.PASTA_PROJETO_DRIVE):
        raise RuntimeError(
            "O Drive não está montado: monte-o no notebook antes de lançar os treinos."
        )

    iniciar_execucao()

    conjuntos = preparar_dados(imagens_no_disco=True)

    treino = treinar(conjuntos)

    avaliar_e_registrar(treino, conjuntos)

    print("Treino concluído:", configuracao.NOME_EXPERIMENTO)


if __name__ == "__main__":
    executar_experimento()
