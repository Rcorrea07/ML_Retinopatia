# Execução da campanha de modelos leves

## Estado confirmado em 20/10/2026

Código publicado na branch `codex/campanha-retinopatia`. A revisão `5abe071`
iniciou ensaio sintético em uma **Tesla T4**, com precisão mista, batch 4,
40 imagens sintéticas e duas épocas (uma por fase). Os ensaios não usam
ImageNet nem fotos do projeto e **não medem desempenho de retinopatia**.

- [Notebook de ensaio ativo no Chrome](https://colab.research.google.com/drive/1fnpmy62CJhij3wsHshKzMZceJb7exlQ2)
- [Verificação criada na conta do projeto, antes da desconexão do Brave](https://colab.research.google.com/drive/1uuCRV2no_gLxGEV9goiqbVMY7IU3Yauy?authuser=4)
- [Referência B0 consultada](https://colab.research.google.com/drive/100PsKg_weebqXZ5XHAmvivG9SLQYlYfh?authuser=4)

O Brave deixou de estar disponível durante a conexão do Drive. O Chrome
está conectado a outra conta do responsável, sem a conta eHealth do projeto.
Não foi confirmado acesso ao dataset nessa sessão, nem iniciados cinco treinos
com dados reais. A autorização do responsável está concedida; a pendência é
a conexão/autenticação da conta que contém o Drive.

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

Se o Colab conceder menos GPUs, manter os mesmos perfis em filas nas GPUs
disponíveis. Não usar várias contas para contornar limites do provedor.
Os notebooks conferem GPU e montagem do Drive antes do treino real. A
preparação dos caches pode demorar; `estado_execucao.json` distingue preparação
de dados de treinamento efetivo.

## Início e verificação

1. Abrir os notebooks na conta com `MyDrive/ProjetoRetinopatia` e
   `MyDrive/Datasets/diabetic-retinopathy-detection.zip`.
2. Selecionar GPU e RAM alta, quando disponível. Conferir sessões existentes
   antes de alocar cinco GPUs, sem interromper trabalho alheio.
3. Fixar `REVISAO_CODIGO` e executar setup + montagem do Drive.
4. Executar os ensaios antes de consumir o dataset completo. E10 sintético
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
