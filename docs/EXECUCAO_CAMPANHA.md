# Execução da campanha de modelos leves

## Estado confirmado em 20/10/2026

Código publicado na branch `codex/campanha-retinopatia`. A revisão `5abe071`
concluiu seis ensaios sintéticos (E01/E02/E03/E09/E10/E11) em uma **Tesla T4**.
E08 concluiu na revisão `c7148d3`. Todos usaram precisão mista, batch 4,
40 imagens sintéticas e duas épocas (uma por fase). Foram conferidos os arquivos
de cada fase, recarga do modelo/otimizador, hashes dos modelos e coerência
das saídas derivadas das cinco classes. Os sete testes locais também passaram. Os ensaios não usam
ImageNet nem fotos do projeto e **não medem desempenho de retinopatia**.

Os cinco notebooks foram criados e salvos pelo Brave na conta
do projeto, com a estrutura original do Drive:

Os links privados dos cinco notebooks estão no relatório local ignorado pelo
git: `relatorios/campanha_2026-10-20/EXECUCAO_PRIVADA.md`.

A montagem segue `ambiente_colab.montar_drive()` do projeto original.
Três sessões receberam GPU T4/RAM alta. As duas primeiras montaram o Drive
e iniciaram E01/E02 com dados reais (estado `preparando_dados`). A configuração
e o código já foram salvos nas pastas de execução. O terceiro Drive também foi montado e E03 começou a preparar os dados.
E01 entrou na primeira época (`treinando_fase1`); nenhum score final foi obtido. O Brave está na conta correta. Os arquivos históricos B0/B4 foram
vistos na pasta `modelos`, inclusive `B4_448_v3_extra_22f95c4.keras` (335,3 MB).
Sua configuração e hash ainda precisam ser conferidos pelo código antes de E10.

A revisão `a6e457a` reserva o espaço do Drive para os artefatos: a conta mostra
8,47 GB usados de 15 GB. Não duplica os ZIPs legados equivalentes. Receitas novas
ficam no `/content` por padrão, com manifestos e tabelas de hashes copiados para
cada execução no Drive. Após perder o runtime será necessário reconstruí-las.
Todos os checkpoints, modelos, históricos e relatórios continuam persistentes.
`RETINOPATIA_PERSISTIR_CACHES_NOVOS=1` permite novos ZIPs quando houver espaço.
O verificador `tests/colab_verificar_cache.py` passou no primeiro runtime
(`returncode=0`). Usou somente imagens sintéticas e pastas temporárias no Colab.


## Evidências locais

O pacote de relatórios foi baixado e copiado para
`relatorios/campanha_2026-10-20/relatorios_ensaios_retinopatia.zip` (ignorado pelo git).
São **187 arquivos**, sem os modelos sintéticos, que continuam no runtime. CRCs
do ZIP e sete registros de conclusão foram conferidos localmente. O manifesto
`verificacao_ensaios.json` e a captura `ensaios_concluidos.jpg` ficam nessa pasta.

SHA256 do ZIP baixado:
`d9e1e7134820acf9e1614bb6c9674cc475aa06a38acfab43fcf4d53200878fba`.
Ambiente registrado: TensorFlow 2.21.0, Keras 3.13.2, Python 3.13.15 e Tesla T4.
**Esses arquivos documentam verificação de software, sem score clínico.**

## Cinco postos, onze configurações

Cada notebook usa um processo novo por perfil e uma única GPU por vez. Fixar
**a mesma revisão imutável** nos cinco postos; não atualizar durante o treino.
As filas terminam um perfil antes de iniciar o seguinte:

| Posto | Fila | Notebook no repositório |
| --- | --- | --- |
| 1 | E01 → E06 → E11 | `notebooks/Campanha_Posto_1.ipynb` |
| 2 | E02 → E07 | `notebooks/Campanha_Posto_2.ipynb` |
| 3 | E03 → E08 | `notebooks/Campanha_Posto_3.ipynb` |
| 4 | E04 → E09 | `notebooks/Campanha_Posto_4.ipynb` |
| 5 | E05 → E10 | `notebooks/Campanha_Posto_5.ipynb` |

Os primeiros cinco são variações controladas da B0. Os seguintes comparam
384 px, GAP, pesos suaves, MobileNetV3Large e destilação. E11 classifica
explicitamente **0/1/2/3/4**: não existe grau 5 no EyePACS. Conserva as saídas
`doente` e `grau`, derivadas da distribuição das cinco classes.

Na execução atual, o Colab recusou a quarta e quinta sessões com **“Sessões em
excesso”**. O gerenciador mostra as três GPUs ocupadas pela campanha. As onze
configurações foram distribuídas nas sessões disponíveis:

| GPU disponível | Fila atual |
| --- | --- |
| Posto 1 | E01 → E06 → E11 |
| Posto 2 | E02 → E07, depois E04 → E09 em célula enfileirada |
| Posto 3 | E03 → E08 → E05 → E10 |

Os postos 4/5 continuam salvos para um cenário com mais sessões, mas não devem
ser iniciados junto dessas filas, pois repetiriam E04/E09 e E05/E10.
Os notebooks conferem GPU e montagem do Drive antes do treino real. A
preparação dos caches pode demorar; `estado_execucao.json` distingue preparação
de dados de treinamento efetivo.

## Início e verificação

1. Abrir os notebooks na conta com `MyDrive/ProjetoRetinopatia` e
   `MyDrive/Datasets/diabetic-retinopathy-detection.zip`.
2. Selecionar GPU e RAM alta, quando disponível. Conferir sessões existentes
   antes de alocar cinco GPUs, sem interromper trabalho alheio.
3. Fixar `REVISAO_CODIGO` e executar setup + montagem do Drive.
4. `notebooks/Verificar_Campanha.ipynb` contém o ensaio reproduzível.
   Executar os ensaios antes de consumir o dataset completo. E10 sintético
   verifica a perda/serialização; o professor real ainda exige auditoria.
5. Iniciar as filas. Conferir `treinando_fase1/2` e crescimento dos históricos
   CSV/checkpoints. Abrir cinco abas não comprova cinco execuções.

Exemplo após setup e montagem do Drive, em notebook que ainda não importou
`configuracao`:

```python
from retinopatia.campanha import executar_fila
resultados = executar_fila(['E01', 'E06', 'E11'], posto='posto1')
```

Ensaios sem Drive são permitidos **apenas com imagens sintéticas**:

```python
import os
os.environ['RETINOPATIA_RAIZ_ENSAIO'] = '/content/retinopatia_ensaios'
from retinopatia.campanha import executar_fila
resultados = executar_fila(['E01', 'E09', 'E10', 'E11'], ensaio=True)
```

Nesse caso, baixar os artefatos antes de encerrar a sessão; `/content` é volátil.

## Artefatos preservados e acrescentados

Cada execução possui perfil, hash da configuração, commit e ID único no nome.
O Drive conserva `configuracao.json`, `codigo_src.zip`, melhores pesos das duas
fases, CSVs com LR/GPU/RAM, predições das duas cabeças, curvas e modelo `.keras`.
Acrescenta snapshots completos com otimizador (`retomada_fase1/2.keras`), JSONs
de estado, métricas/limiares por fase, checksum do modelo e verificação de recarga.
Também salva diagnóstico por grau, comparação de cabeças/agregadores e bootstrap
por paciente; E11 acrescenta cinco probabilidades, matriz 5×5 e QWK.

A análise principal usa passagem única; TTA4 é gravada separadamente. O teste
histórico não é predito nessa campanha. Cada execução escreve seu `resumo.csv`;
um único consolidador atualiza o CSV legado após a conclusão das filas:

```python
from retinopatia.avaliacao import consolidar_experimentos
registro = consolidar_experimentos()
```

As filas gravam JSON e log por execução em
`MyDrive/ProjetoRetinopatia/campanhas/2026-10-20/postoN/`.
Ensaios ficam em `ensaios/` e são excluídos da consolidação de scores.

## Retomada explícita

Repetir uma chamada de fila cria uma execução nova e preserva a anterior.
Para retomar, usar a **mesma revisão, perfil, semente e ID** em processo novo:

```python
import os, subprocess, sys
ambiente = dict(os.environ, RETINOPATIA_PERFIL='E01',
                RETINOPATIA_ID_EXECUCAO='ID_COPIADO_DO_REGISTRO',
                RETINOPATIA_RETOMAR='1')
subprocess.run([sys.executable, '-m', 'retinopatia.campanha', '--perfil', 'E01'],
               env=ambiente, check=True)
```

Código/configuração incompatíveis são recusados. A retomada restaura época,
otimizador, scheduler e espera/melhor métrica do early stopping; ela não promete
reproduzir bit a bit a ordem aleatória de augmentation/shuffle após desconexão.

## Dados e limites da auditoria

Split v2/42 preservado e conferido por paciente, imagem e rótulo; seed de treino
separada. Caches têm hash da receita, manifesto, contagem, dimensão e tabela de
hashes. Restauração compara CRC do JPEG; extração interrompida não é considerada
completa só porque a pasta contém algum arquivo.

Cópias de pixels processados entre treino e conjuntos retidos são removidas do
treino com mapa de exclusões. Correspondências exatas de pHash e conflitos de
rótulo são registrados para revisão. Isso **não demonstra ausência de duplicatas
nas fotos brutas nem cobre toda semelhança perceptual**. Exclusões diferentes
entre receitas precisam ser consideradas ao comparar resultados.

E10 só aceita a B4 histórica esperada, com configuração e arquivos verificáveis,
e recusa professor potencialmente exposto a cópias dos retidos. Professor gera
alvos offline a 448 px, sem TTA, somente no treino; aluno a 384 px usa augmentation.
Esse professor não faz parte do modelo entregue à aplicação.

Após os treinos: consolidar, comparar validação, repetir referência e dois
finalistas com seeds 17/73 sem mudar split e medir custo de inferência no servidor.
A API/PWA e exportação local ainda exigem trabalho posterior. Não há ganho de
score confirmado até obter as execuções reais e comparar no protocolo definido.
