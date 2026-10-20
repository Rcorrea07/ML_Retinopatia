# Referências para os experimentos

Consulta em 20/10/2026. Kaggle e Colab abertos no Brave indicado pelo usuário.
Este registro distingue leitura confirmada, score oficial, código localizado e
acesso parcial. Reconsultar os links relevantes ao implementar cada perfil.
Os identificadores F01–F18 também aparecem no catálogo JSON e no plano.

## Scores oficiais EyePACS 2015

Métrica: kappa com pesos quadráticos, cinco graus por olho. Não equivale à
ROC-AUC binária nem à especificidade por paciente do projeto. Os 53.576 casos do
teste da competição agora integram o treino ampliado do projeto; depois disso,
não se pode apresentar um score nesses casos como generalização ao teste Kaggle.

| Posição final | Equipe | Público (F02) | Privado final (F01) | Código |
| --- | --- | ---: | ---: | --- |
| 1 | Min-Pooling | 0,86035 | 0,84957 | F12, baixado |
| 2 | o_O | 0,85425 | 0,84478 | F13, baixado |
| 3 | Reformed Gamblers | 0,85106 | 0,83936 | ZIP anexo a F05, baixado |
| 4 | Julian de Wit & Daniel Hammack | 0,85105 | 0,83625 | Não localizado nesta consulta |
| 5 | Jeffrey De Fauw | 0,84202 | 0,82898 | F14, baixado |

F01: [leaderboard final](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/leaderboard).
F02: [leaderboard público](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/leaderboard?tab=public).
A interface informa aproximadamente 20% dos dados para público e 80% para
privado. O PDF de Ben informa 0,84958; a tabela oficial consultada mostra 0,84957.
Preservar essa pequena divergência; não ajustar o número oficial pelo relatório.

## O que aproveitar de cada solução

**F03 — [Min-Pooling / Ben Graham](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/writeups/min-pooling-competition-report-min-pooling-and-tha).**
Relatório e script original lidos. Normaliza o raio, subtrai média local de cor,
mantém 90% do raio e usa rotações. O script define sigma como raio/30. O autor
relata que transformações de cor adicionais não ajudaram. Combina olhos com
Random Forest, mas reconhece o risco de treinar esse combinador em predições de
treino da CNN. Adaptação: E03/E04 e pré-processamento consistente, sem copiar o
empilhamento com essa fragilidade.

**F04 — [Team o_O](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/writeups/o-o-team-o-o-solution-summary).**
Texto e arquivos selecionados lidos. Recorta fundo, treina progressivamente em
resoluções maiores, usa regressão, balanceamento que diminui no decorrer do
treino e combina informação dos dois olhos. A implementação usa Python 2,
Theano/Lasagne. Adaptação: comparar balanceamento e explorar resolução;
transferir ideias ao pacote atual. O procedimento original de dezenas de TTA
e múltiplos modelos não define o custo desejado para inferência.

**F05 — [Reformed Gamblers](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/writeups/reformed-gamblers-3rd-place-solution-report).**
Resumo, PDF v2 e READMEs do ZIP lidos. Nove CNNs, perdas de regressão,
augmentation na GPU e combinação dos olhos. O relatório compara imagens de
385 e 767 px com kappa de validação 0,808 e 0,843; não é score de teste. Resolução
maior ainda deixou de ajudar. Adaptação: medir resolução da B0 e manter o
augmentation atual na GPU; as CNNs Torch7/CUDA originais servem como referência.

**F06 — [Guanshuo Xu, APTOS 1º lugar](https://www.kaggle.com/competitions/aptos2019-blindness-detection/writeups/guanshuo-xu-1st-place-solution-summary).**
Texto e trecho de GeM lidos. Relata ensemble de oito modelos, SmoothL1, GeM e
redimensionamento simples. Informa kappa privado de aproximadamente 0,936 após
dados adicionais/pseudo-rótulos e limiares. Isso é relato do autor, não nova
medição nossa. Usou leaderboard público para validação e pseudo-rótulos de teste.
Adaptação: ablação de Ben e pooling, preservando avaliação independente. Link
de código GeM encontrado: [cnn retrieval PyTorch](https://github.com/filipradenovic/cnnimageretrieval-pytorch);
o repositório completo não foi auditado ou baixado nesta consulta.

**F07 — [ODS.ai, APTOS 4º / 2º após limpeza](https://www.kaggle.com/competitions/aptos2019-blindness-detection/writeups/ods-ai-eye-of-private-lb-4th-place-solution-2nd-af).**
Texto e comentários técnicos dos autores lidos. A página ainda usa “4th” no
título, mas o metadado da posição e a discussão confirmam 2º após a limpeza.
Usa B3/B4/B5, dados de 2015, recorte ou simples resize e TTA de espelhamento;
relata ausência de ganho com Ben e uso de pseudo-rótulos. Adaptação: testar o
pré-processamento em B0. Não foi localizado repositório da solução nessa página;
não atribuir a ela um código de terceiros ou um score numérico não conferido.

**F15 — [Jeffrey De Fauw, 5º lugar](https://defauw.ai/diabetic-retinopathy-detection/).**
Fonte adicional descoberta pelo leaderboard e conferida com repositório do
autor. Há combinação de representações dos olhos, preocupação com detalhes
finos e redução do oversampling para voltar à distribuição real. README,
licença e perdas do código foram lidos; o post foi consultado nas seções
introdutórias e de arquitetura. E08 usa isso como motivação para uma hipótese
mais simples; não reproduz a solução original.

## Notebooks pedidos: acesso parcial

**F08 — [Meena Vyas](https://www.kaggle.com/code/meenavyas/diabetic-retinopathy-detection).**
Página aberta, versão 12/12, licença Apache 2.0. O iframe do código ficou vazio.
O download oferecido pela página levou a
`https://www.kaggle.com/kernels/scriptcontent/4781533/download`, bloqueado pelo
Brave com `ERR_BLOCKED_BY_CLIENT`. Não foi alterada a proteção do navegador.
Código e score não foram verificados; reabrir na execução. Votos/medalha não
demonstram desempenho de validação.

**F09 — [K Scott Mader / InceptionV3](https://www.kaggle.com/code/kmader/inceptionv3-for-retinopathy-gpu-hr).**
Página aberta, versão 6/6, licença Apache 2.0 e referência a execução P100. O
conteúdo do notebook não ficou disponível no render consultado. Código, split
e métricas precisam de leitura antes de aproveitamento. InceptionV3 não entra
como modelo padrão de implantação nesta proposta.

## Repositórios localizados e cópias

| Fonte | Origem | Revisão baixada | Material conferido |
| --- | --- | --- | --- |
| F12 | [SparseConvNet arquivado, branch da competição](https://github.com/btgraham/SparseConvNet-archived/tree/kaggle_Diabetic_Retinopathy_competition) | `a6bdb0c938b3556c1e6c23d5a014db9f404502b9` | Pré-processamento e combinação; GPLv3 indicada no README |
| F13 | [sveitser/kaggle_diabetic](https://github.com/sveitser/kaggle_diabetic) | `e6f1cf3109eaf87db12d77d48f73e62a084fd10a` | README, outline, recorte e licença MIT |
| F14 | [JeffreyDF/kaggle_diabetic_retinopathy](https://github.com/JeffreyDF/kaggle_diabetic_retinopathy) | `77c57811ad6386ad0548f2b755e0536c1d652774` | README, perdas, licença MIT; ZIP inclui artefatos do autor |
| F05 | [ZIP de submissão Reformed Gamblers](https://storage.googleapis.com/kaggle-forum-message-attachments/88866/2813/rfg_submission_code.zip) | ZIP fixo do anexo; SHA-256 no catálogo | READMEs, estrutura e licença incluída; reutilização ainda exige verificar arquivos incorporados |

O [link antigo de Ben](https://github.com/btgraham/SparseConvNet/tree/kaggle_Diabetic_Retinopathy_competition)
retornou 404. O README do repositório atual apontou o arquivo histórico, onde a
branch foi confirmada no Brave. Não confundir o SparseConvNet atual com o código
da competição de 2015.

Arquivos estão em `relatorios/referencias/2026-10-20/`, pasta já ignorada pelo Git.
O [catálogo JSON](referencias_retinopatia.json) versiona URLs, commits, tamanhos e
SHA-256. Os ZIPs foram conferidos sem executar seus scripts ou carregar seus
pesos. O plano propõe implementação em Keras no projeto atual.

Relatórios originais preservados:

- [Ben Graham, PDF](https://storage.googleapis.com/kaggle-forum-message-attachments/88655/2795/competitionreport.pdf).
- [Reformed Gamblers, PDF v2](https://storage.googleapis.com/kaggle-forum-message-attachments/88866/2815/Team_Reformed_Gamblers_Solution_Summary_v2.pdf).

## Execução e documentação técnica

- F10: [Colab na conta autorizada](https://colab.research.google.com/?authuser=4).
- F11: [notebook B0 consultado](https://colab.research.google.com/drive/100PsKg_weebqXZ5XHAmvivG9SLQYlYfh?authuser=4).
- F16: [FAQ oficial Colab](https://research.google.com/colaboratory/faq.html),
  consultada: recursos e limites variáveis; cinco sessões ainda não demonstradas.
- F17: [MobileNetV3 no Keras](https://keras.io/api/applications/mobilenet/mobilenet_models/),
  consultada: entrada 0–255 com pré-processamento incluído; o benchmark ImageNet
  da página não representa score de retinopatia nem latência no nosso aparelho.
- F18: [destilação, exemplo oficial Keras](https://keras.io/examples/vision/knowledge_distillation/)
  e [código oficial](https://github.com/keras-team/keras-io/blob/master/examples/vision/knowledge_distillation.py).
  Página retornou falha de carregamento na abertura; descrição foi localizada
  na busca oficial. Ler o código completo antes de implementar E10. A mistura
  binária 0,7/0,3 do plano é hipótese própria, não receita validada desse exemplo.

## Rotina de reconsulta durante a execução

1. Abrir a fonte e os arquivos originais usados pelo perfil; verificar versão,
   métrica, conjunto de avaliação e se há teste usado para seleção/pseudo-rótulos.
2. Consultar o código fixado pelo commit e registrar o trecho que será adaptado.
3. Conferir licença/atribuição antes de incorporar código; não instalar o stack
   legado para transportar apenas uma ideia de pré-processamento ou treino.
4. Salvar em cada execução os IDs das fontes consultadas, data e decisões.
5. Se o código contradizer este resumo, registrar a diferença e atualizar o
   perfil antes de treiná-lo. Não tratar este planejamento como evidência final.
