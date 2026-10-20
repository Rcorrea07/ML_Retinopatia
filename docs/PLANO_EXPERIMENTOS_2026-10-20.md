# Plano de experimentos de retinopatia

Data: 20/10/2026. Código auditado: `22f95c4`.

Estado: perfis E01–E11 implementados e publicados na branch
`codex/campanha-retinopatia`. Sete ensaios sintéticos concluídos em uma GPU T4
no Colab; treinamento com fotos reais ainda não confirmado. A sessão do Brave
foi desconectada durante a montagem do Drive. A conta do projeto não está
autenticada no Chrome disponível. As hipóteses não representam ganhos medidos.

Ver [guia operacional e execução](EXECUCAO_CAMPANHA.md) para filas, retomada,
artefatos e estados confirmados. Este plano continua registrando a intenção;
os arquivos de cada execução registram o que realmente aconteceu.

## Objetivo e escopo

Melhorar a triagem binária de retinopatia, preservando a cabeça auxiliar de grau
e todos os registros existentes. A prioridade confirmada pelo responsável é
**PWA com inferência no servidor primeiro; execução no próprio celular depois**.
O usuário do PWA deve continuar recebendo apoio à triagem de retinografias.

O repositório atual contém treinamento e avaliação. A aplicação Next.js/FastAPI
é descrita em documentos locais, mas seu código não está neste checkout. A
documentação de integração de julho já descreve um contrato binário; o contexto
arquitetural inicial, com cinco classes, é anterior a essa integração. A API
efetivamente em uso deverá ser inspecionada antes de trocar o modelo.

EfficientNetB0 continua como candidata principal: tem aproximadamente 4,05 milhões
de parâmetros neste projeto. B4 já treinada pode ensinar a B0 por destilação, sem
colocar o professor no produto. Evitar aumentar o backbone por padrão. Resolução,
pré-processamento, TTA, memória e latência também entram na decisão de implantação.

## Referência atual, confirmada no Colab

Foi aberto no Brave e baixado o [notebook B0 com as saídas](https://colab.research.google.com/drive/100PsKg_weebqXZ5XHAmvivG9SLQYlYfh?authuser=4).
A cópia local está em
`relatorios/referencias/2026-10-20/Retinopatia_treino_B0_consultado.ipynb`.

- Código: `22f95c4`; B0, 448 px, pré-processamento v3, semente 42.
- Treino: 81.674 imagens = 28.098 originais + 53.576 extras do EyePACS 2015.
- Validação e teste: cada um com 3.514 olhos / 1.757 pacientes.
- Melhor época da fase 2: 11; `val_doente_pr_auc` do Keras = 0,8614.
- Avaliação final: TTA com quatro variantes; regra por paciente = máximo dos olhos.

| Conjunto / unidade | ROC-AUC | AP (chamada PR-AUC no relatório) | Sensibilidade | Especificidade | Limiar |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validação / olho | 0,9126 | 0,8702 | 0,9003 | 0,6676 | 0,1944 |
| Validação / paciente | 0,9098 | 0,8788 | 0,9002 | 0,6867 | 0,2569 |
| Teste histórico / olho | 0,9171 | 0,8784 | 0,9034 | 0,6598 | 0,1944 |
| Teste histórico / paciente | 0,9042 | 0,8799 | 0,8848 | 0,6851 | 0,2569 |

O registro de experimentos visível nessa mesma saída contém também B4 com dados
extras: teste por paciente ROC-AUC 0,9232, sensibilidade 0,8976 e especificidade
0,6926. Isso é evidência histórica; a disponibilidade e a identidade do arquivo
`B4_448_v3_extra_22f95c4.keras` ainda precisam ser verificadas no Drive para E10.
O diagnóstico final do notebook B0 baixado não contém saída salva.

`tf.keras.metrics.AUC(curve="PR")` durante o treino e
`average_precision_score` na avaliação são estimadores diferentes. TTA também
muda as predições. Não comparar 0,8614 e 0,8702 como se fossem a mesma medição.

## Auditoria: pontos a resolver antes dos cinco treinos

| Achado no código | Consequência | Trabalho necessário |
| --- | --- | --- |
| `SEED` controla pesos, augmentation, shuffle e `StratifiedGroupKFold` | Trocar a semente pode trocar pacientes e invalidar comparação pareada | Separar `SEED_SPLIT=42` de `SEED_TREINO`; reutilizar os CSVs originais com checksums |
| `NOME_EXPERIMENTO` não inclui todos os hiperparâmetros e reutiliza a pasta na repetição | Duas configurações ou reexecuções podem sobrescrever checkpoints, configuração e CSVs | Identificador do perfil + hash da configuração resolvida + identificador único de execução; recusar pasta existente salvo retomada explícita |
| `NOME_MODELO` não contém semente nem perfil completo | Modelos de execuções distintas podem disputar o mesmo `.keras` | Usar o mesmo identificador único nos modelos e nos resultados |
| `registrar_experimento` lê e regrava um CSV compartilhado | Cinco sessões podem perder linhas por concorrência | Cada execução escreve seu resumo; um único consolidador atualiza o CSV legado depois |
| Notebook avalia teste e roda diagnóstico ao executar tudo | Uma busca de dez versões expõe teste repetidamente | Modo exploração sem predições/métricas de teste; registro aceita só validação; diagnóstico separado por conjunto |
| Cache identificado sobretudo por versão e tamanho; lista suspeita só por versão | Alterações em sigma, raio ou máscara podem reutilizar material errado | Manifesto e hash de todos os parâmetros de imagem, resolução, origem e contagem; versões novas sem apagar v2/v3 |
| Restauração verifica pasta não vazia, sem manifesto completo | Extração interrompida pode parecer pronta | Validar integridade e imagens esperadas; preparar caches uma vez, publicar completos e reutilizar como somente leitura |
| Defaults das funções são ligados na importação de `configuracao` | Mudar constantes depois dos imports deixa valores/caminhos antigos | Resolver perfil antes de importar o pacote; reiniciar sessão ao trocar perfil; congelar configuração durante execução |
| `salvar_configuracao` salva constantes e `codigo_src.zip` | Uma mudança só em argumento de função pode não aparecer na configuração | Registrar configuração efetivamente aplicada, argumentos e manifesto do perfil; preservar o zip atual |
| Pacientes extras são comparados por ID; duplicatas visuais não são verificadas | IDs diferentes não demonstram ausência de duplicação | Auditar hashes de JPEG/pixels processados e suspeitas por pHash; auditar fotos brutas e duplicatas próximas antes de declarar ausência de vazamento |
| `processar_retina(aplicar_ben=False)` ainda usa defaults globais de cor | Passar apenas esse argumento pode deixar fundo incompatível | Resolver fundo e augmentation juntos: sem Ben = 0, com Ben = 128; cache próprio |
| `GeM` aplica potência a ativações limitadas por epsilon | Outra arquitetura pode ter distribuição diferente; exportação não está testada | Comparar GAP, preservar carregamento de GeM e verificar conversão; MobileNet usa GAP nesta campanha |
| A documentação antiga atribui um teto aos dados e afirma ausência de vazamento | São conclusões mais fortes que a evidência disponível | Tratar como hipóteses; auditar dados e repetir finalistas sem escolher pela métrica de teste |

Não há evidência aqui de que os pesos tenham treinado nos pacientes do teste
interno. Há exposição repetida às métricas e aos erros desse teste. Proximidade
entre scores de validação e teste, sozinha, não prova ausência de vazamento.

## Protocolo contra viés

1. Preservar os splits v2/42 e os resultados históricos. Se a auditoria revelar
   duplicatas entre conjuntos, versionar uma correção com mapa de exclusões e
   marcar a quebra de comparabilidade; não trocar pacientes silenciosamente.
2. Usar validação para checkpoint, comparação, limiar e TTA. Nos dez perfis,
   manter o conjunto de teste fora da execução de exploração.
3. Congelar por escrito perfis, métrica principal, orçamento, regra de desempate
   e política de imagens ruins antes da primeira GPU. Este arquivo é uma proposta;
   a configuração resolvida da execução será a fonte do que realmente rodou.
4. Comparação principal operacional: **um modelo, uma passagem por olho, máximo
   por paciente**, limiar escolhido para sensibilidade empírica de pelo menos 0,90
   na validação. Classificar pela especificidade nesse ponto. Reportar também
   ROC-AUC, AP, matriz, sensibilidade por grau e alvo referenciável separadamente.
5. Medir TTA de quatro variantes como análise secundária em todos os candidatos,
   com predições/limiares identificados. Manter comparador histórico com TTA; nunca
   aplicar um limiar obtido com TTA a uma inferência sem TTA.
6. Separar, por paciente, seleção e calibração quando o tamanho das classes
   permitir, ou usar predições fora da dobra para comparar/ajustar agregadores.
   Qualquer ajuste aprendido para combinar olhos usa predições fora da amostra.
   O máximo permanece a regra da campanha inicial; média é análise secundária.
7. Repetir a referência e os dois finalistas com sementes de treino 17 e 73,
   mantendo split 42. São execuções adicionais às dez configurações. Reportar
   média, dispersão e diferenças pareadas por paciente. Bootstrap da mesma
   validação não remove o otimismo de selecionar o melhor entre dez.
8. O teste antigo serve de benchmark histórico já consultado. Para nova estimativa
   independente, reservar uma fonte externa que não tenha sido usada em treino
   nem seleção: APTOS rotulado é candidato, condicionado a procedência, permissões
   de uso e auditoria de duplicatas. Nesta campanha, não usá-lo no treino.
   Resultado externo mede também mudança de domínio; não substitui validação
   na população real. Não chamar uma nova divisão de dados já vistos de teste cego.
9. Congelar modelo, formato de inferência, regra e limiar antes da avaliação
   externa. Reportar a sensibilidade obtida e seu intervalo; escolher 90% na
   validação não garante 90% em novos dados.

## Onze configurações implementadas

Cada perfil herda a referência, salvo alterações explícitas. Comparações são
hipóteses, sem promessa de superar o score. Valores completos estão no
[manifesto dos experimentos](experimentos_2026-10-20.json). O módulo
`campanha` lê a cópia versionada `src/retinopatia/perfis_experimentos.json`.

Referência comum: B0/ImageNet, 448 px, GeM p=3, dropout 0,4, BCE + 0,5 Huber
(delta 1), pesos binários balanceados nas duas cabeças, Adam, fase 1 de 2 épocas
a 1e-3; fase 2 de até 12 épocas, cosseno com aquecimento de uma época, máximo
5e-5 e fração final 0,01. Batch 32 inicial. BN congelada. Mesmos dados extras.

| ID | Leva | Alteração frente ao perfil indicado | Pergunta respondida | Fontes |
| --- | ---: | --- | --- | --- |
| E01 | 1 | Referência B0 448 v3 | Reproduz o resultado com o protocolo e registros novos? | Código atual / F11 |
| E02 | 1 | E01 sem Ben; mesmo recorte/máscara, fundo e preenchimento 0 | Preservar cor original ajuda mais que normalizar? | F06, F07 |
| E03 | 1 | E01 com sigma de Ben = raio normalizado / 30 = 448/60 | O sigma proporcional funciona melhor que o 10 fixo? | F03, F12 |
| E04 | 1 | E01 com brilho/contraste 0,03 em vez de 0,10; geometria igual | A alteração de cor está forte demais após Ben? | F03 |
| E05 | 1 | E01 com peso de Huber 0,20 em vez de 0,50 | A tarefa auxiliar está competindo com a triagem binária? | F06 + hipótese do projeto |
| E06 | 2 | E01 a 384 px; demais parâmetros iguais, inclusive sigma 10 | Qual a troca entre lesões pequenas, score e custo? | F03, F05, F07 |
| E07 | 2 | E01 com GlobalAveragePooling2D no lugar de GeM | Quanto GeM entrega e quanto simplificar ajuda a exportar? | F06 |
| E08 | 2 | E01 com raiz dos pesos balanceados, renormalizada para média 1 no treino | Balanceamento mais suave reduz falsos positivos? | F04, F15 + hipótese do projeto |
| E09 | 2 | MobileNetV3Large alpha 1,0, 384 px, GAP; resto de E06 | Outra arquitetura leve oferece melhor custo/desempenho? | F17 |
| E10 | 2 | E06 com destilação binária de B4 extra já treinada | O aluno de 384 px recupera parte do desempenho do professor? | F18 |
| E11 | 2 | E01 com softmax de cinco graus, pesos balanceados por grau e triagem derivada de P(grau ≥ 1) | A classificação explícita dos graus ajuda a triagem? | Pedido do responsável / hipótese do projeto |

E03 é adaptação da proporção raio/30; mantém a máscara v3 e não reproduz
integralmente a solução de Ben Graham. E06 isola a resolução dentro da receita
atual; não muda sigma ou erosão junto. E08 altera os mesmos pesos aplicados às
duas saídas, como hoje, e não combina oversampling com peso extra.

E09 também muda pooling para se ajustar à arquitetura, portanto é comparação de
pacote de modelo com E06/E07, sem atribuir causalmente todo ganho ao backbone.
`include_preprocessing=True` mantém entrada 0–255, conforme Keras; nunca adicionar
outro rescale por fora. Os backbones existentes permanecem disponíveis.

Em E10, professor congelado e auditado contra os mesmos pacientes retidos; sem
usar teste/validação como dados de destilação. Proposta inicial: temperatura 1,
perda binária = 0,7 BCE(rótulo real) + 0,3 BCE(probabilidade do professor), mais
0,5 Huber(grau real). **Adaptação implementada:** o professor vê a versão
original 448 px e gera alvos offline sem TTA; o aluno vê a versão 384 px com
augmentation. Não há transformação geométrica compartilhada entre ambos. Reutilizar somente um professor para manter
o orçamento; o arquivo dele e seu pré-processamento recebem checksum. Se não
for verificável, E10 fica pendente; não trocar por outro experimento sem registrar.

As ideias de perda ordinal, CLAHE, focal loss, resolução progressiva, ensembles
e dados adicionais ficam como próximas hipóteses. Não acumular mudanças antes
de saber o efeito dessas dez. Combinar as melhorias confirmadas será uma rodada
posterior declaradamente adaptativa, novamente escolhida sem o teste histórico.

E11 utiliza os graus **0, 1, 2, 3 e 4** do EyePACS. Não existe classe 5
nesses rótulos. `doente` é a soma das probabilidades 1–4 e `grau` é o valor
esperado; a interface das saídas antigas é preservada. São gravadas também
cinco probabilidades, classe argmax, matriz 5×5, métricas por grau e QWK.

## Como executar cinco ao mesmo tempo no Colab

**Um GitHub é suficiente.** Usar um único pacote com perfis de configuração e
cinco cópias finas do notebook, cada uma em uma sessão de execução independente.
Não fazer cinco cópias da lógica Python ou abrir cinco processos competindo pela
mesma GPU. Ter cinco abas não demonstra ter cinco GPUs.

1. Implementar perfis e verificações, publicar uma revisão imutável no repositório
   existente e fixá-la nas cinco sessões. Alterações locais só chegam ao Colab
   depois de publicadas ou transferidas explicitamente.
2. Preparar e verificar os caches antes da leva, uma vez por perfil de imagem.
   Copiar/extrair para `/content` em cada sessão. Não pré-processar todos os
   perfis simultaneamente lendo milhares de arquivos individuais do Drive.
3. Criar notebooks `campanha_E01` a `campanha_E05`, configuração antes dos imports,
   com perfil, semente de treino, revisão e modo exploração. Cada chamada longa
   fica em módulo; o notebook continua orquestrador.
4. Conferir conta, sessões já existentes, unidades disponíveis, GPU/RAM/disco e
   independência dos ambientes. Usar a conta autorizada. Não comprar recursos
   nem assumir simultaneidade com base no rótulo Pro+.
5. Fazer ensaio curto com amostras de treino/validação: leitura, pesos nas duas
   cabeças, checkpoint, recarga, predição, logs e persistência. Registrar o ensaio
   separadamente; ele não conta como resultado de E01.
6. Iniciar E01–E05 com até cinco sessões, se o Colab disponibilizar. Confirmar
   progresso real dos históricos/checkpoints de cada sessão. Se houver menos
   sessões disponíveis, rodar uma fila com a mesma lista e configurações.
7. Depois, usar os mesmos cinco postos para E06–E10. Professor de E10 pode exigir
   lote menor; mudança registrada, inclusive efeito sobre passos do cosseno.
8. Cada sessão grava só na sua pasta; ao final, um consolidador lê todos os
   resumos e atualiza o registro geral. Não fazer `git pull` durante treino.

A [FAQ oficial do Colab](https://research.google.com/colaboratory/faq.html)
confirma limites dinâmicos e disponibilidade não garantida, inclusive nas opções
pagas. Cinco sessões simultâneas ainda não foram alocadas/verificadas nesta
pesquisa. O notebook B0 estava desconectado; sua interface oferecia conexão T4.

## Contrato de preservação e acréscimos

| Preservar por execução | Acrescentar sem remover o existente |
| --- | --- |
| `configuracao.json` com constantes, commit, alterações locais, GPU e TF | Perfil resolvido, seed de split/treino, UUID, hashes de dados e parâmetros |
| `codigo_src.zip` | Manifesto da campanha, notebook, versões Python/Keras/CUDA e bibliotecas |
| Melhores pesos das fases 1 e 2 | Estado de retomada com otimizador, época e scheduler; pesos sozinhos não retomam exatamente |
| Históricos CSV, LR, pico GPU e RAM | Estado iniciado/em execução/concluído/falhou e motivo da falha |
| Predições por imagem nas duas cabeças | Identificação de passagem única/TTA e conjunto; validação em todos, teste só no final |
| Modelo final `.keras` | Hash do artefato, modelo de inferência e metadados de pré/pós-processamento |
| Registro de experimentos | Resumo individual e consolidação por escritor único; manter CSV legado |
| Splits e caches versionados | Verificação de integridade, origem, resolução e lista de suspeitas por perfil |
| Diagnóstico por grau, referenciável, cabeça, paciente e bootstrap | Salvar tabelas/curvas/limiares em arquivos, além das saídas do notebook |

O resumo tabular não deve exigir resultado de teste para registrar um treino
exploratório. Ao final, campos de teste ficam vazios quando não avaliados.
Novos formatos de exportação não substituem os arquivos de treino antigos.

## Entregas de implementação, na ordem

- [ ] Reconsultar fontes F01–F18 de interesse, registrando data e divergências.
- [ ] Recuperar do Drive configuração e predições B0/B4 extra; confirmar checksums.
- [ ] Auditar dados/splits e fixar protocolo de seleção/calibração.
- [x] Implementar configuração por perfil, sementes separadas e ID sem colisões.
- [x] Implementar resumo individual, estado de execução e retomada sem perda.
- [x] Implementar modo exploração sem teste e persistência completa das análises.
- [x] Implementar perfis de pré-processamento e cache validado, sem apagar v3.
- [x] Implementar GAP, pesos suaves, MobileNet e destilação com saídas compatíveis.
- [ ] Verificar estaticamente Python/notebooks; verificar execução e recarga no Colab.
- [x] Preparar e publicar cinco notebooks finos. Conferência das cinco GPUs ainda pendente.
- [ ] Executar duas levas, consolidar resultados e repetir referência/finalistas.
- [ ] Medir inferência do modelo de passagem única na FastAPI em CPU: p50/p95,
      RAM, tempo de pré-processamento e resposta por exame de um/dois olhos.
- [ ] Escolher o menor candidato na fronteira de score e custo; publicar no PWA
      com limiar e pré-processamento correspondentes, depois de verificar o código
      da aplicação e comparar probabilidades com o Colab nas mesmas imagens.
- [ ] Preparar exportação LiteRT/TFLite float32/float16/int8 e medir paridade;
      calibração int8 apenas com imagens de treino. Decisão móvel/embarcada depois
      depende de medição no aparelho e operações realmente suportadas.

Critério de triagem inicial: superar a especificidade da nova execução E01 na
validação com a sensibilidade alvo, sem deterioração material dos graus graves.
Uma diferença de 0,01 na ROC-AUC e ganho de 0,02 na especificidade são referências
de relevância para discussão, não garantias nem critérios clínicos validados.
Repetições e intervalos pareados decidem se o ganho é estável. O formato de
inferência implantado precisa de limiar próprio e teste de paridade.

Não estimar latência móvel pelo tamanho do `.keras`, nem usar os tempos antigos
de A100 para prometer duração dos novos treinos. O histórico B0 extra mostra
aproximadamente 380 s por época de fine-tuning, mas hardware/versões de cada nova
sessão precisam ser registrados. Nenhuma biblioteca de ML foi instalada localmente.

## Fontes e continuidade

Consultar [referências e achados](REFERENCIAS_RETINOPATIA.md) e
[catálogo de URLs, estados de acesso e downloads](referencias_retinopatia.json).
Eles preservam os links recebidos, os repositórios descobertos, links corrigidos
e o que ainda precisa ser lido. **Reabrir as fontes durante a implementação**:
este plano não substitui código original, relatório ou evidência de execução.
