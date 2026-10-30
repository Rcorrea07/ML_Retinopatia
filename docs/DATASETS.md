# Datasets de retinopatia além do EyePACS

Pesquisa de 30/10/2026, para decidir o que acrescentar ao treino e o que
guardar para a validação externa. Duas regras guiam tudo:

1. **O que entra no treino não serve mais para validação externa.** A prova de
   que o modelo funciona fora do EyePACS precisa de fotos que ele nunca viu.
2. **Nada derivado do EyePACS entra.** Cópias redimensionadas, filtradas ou
   "combinadas" repetem as nossas fotos, inclusive as da validação e do teste,
   e inflariam os resultados.

A validação e o teste do projeto continuam os do EyePACS, para comparar com as
rodadas anteriores. O teste (R09) só abre no fim, com o modelo definitivo.

## Recomendação

| Papel | Dataset | Fotos | Origem e câmera | Rótulos | Acesso | Prioridade |
|---|---|---:|---|---|---|---|
| Treino | **DDR** | 13.673 (~12.500 graduáveis) | China, 147 hospitais; Topcon, Nikon, Canon | ICDR 0–4, mais uma classe "não graduável" (descartar) | Aberto: GitHub (links do Google Drive e Baidu); espelho no Kaggle, já recortado e sem as não graduáveis | 1 |
| Treino | **APTOS 2019** | 3.662 com rótulo | Índia (Aravind Eye Hospital) | ICDR 0–4 | Kaggle (competição); o usuário já tem | 1 |
| Treino | **BRSET** | 16.266 (8.524 pacientes) | Brasil, São Paulo; Canon CR2 e Nikon NF5050 | ICDR 0–4 e SDRG, mais outras doenças | PhysioNet, acesso credenciado (curso CITI e termo de uso) | 2 |
| Treino | **DeepDRiD** | 2.000 (500 pacientes, 2 fotos por olho) | China (Xangai) | ICDR 0–4 e qualidade da foto | Aberto: GitHub | 2 |
| Treino | **IDRiD** | 516 | Índia; Kowa VX-10α | ICDR 0–4, edema macular; máscaras de lesão em 81 | Aberto (CC BY 4.0): IEEE DataPort ou grand-challenge | 3 |
| Treino | **FGADR** | 2.842 (1.000 só com grau) | — | Grau por 6 oftalmologistas; lesões em 1.842 | Pedido aos autores | 3 |
| Validação externa | **Messidor-2** | 1.748 (874 exames, os 2 olhos) | França; Topcon TRC NW6 | ICDR 0–4 adjudicado por 3 retinólogos (rótulos do Google, no Kaggle) | Fotos: pedido no site da ADCIS (há cópia recortada no Kaggle); rótulos: Kaggle | 1 |
| Validação externa | **mBRSET** | 5.164 (1.291 pacientes) | Brasil (Itabuna, BA); câmera portátil Phelcom Eyer acoplada a celular | ICDR 0–4 e edema, adjudicados | PhysioNet, acesso credenciado | 1 |

Por que essa divisão:

- **DDR** é o maior ganho: quase 13 mil fotos novas, de câmeras e hospitais que
  o modelo nunca viu. **APTOS** já está baixado.
- **BRSET** traz a população brasileira, o público do app. Vai para o treino,
  e o mBRSET, do mesmo grupo de pesquisa, fica para a prova.
- **Messidor-2** é a validação externa mais usada nos artigos de retinopatia,
  com rótulos de consenso de especialistas e os dois olhos por exame: dá para
  medir por paciente, como fazemos no EyePACS.
- **mBRSET** é o mais parecido com o uso real do app: fotos de câmera portátil
  de celular, tiradas numa campanha de diabetes no Brasil. Mostra se o modelo
  aguenta fotos piores que as de consultório.
- DeepDRiD, IDRiD e FGADR são pequenos: entram se der pouco trabalho.

Somando DDR, APTOS, BRSET, DeepDRiD e IDRiD, o treino passa de ~82 mil para
~117 mil fotos (+43%), e o tempo de cada treino cresce na mesma proporção.

## Não usar

- **Derivados do EyePACS:** EyeQ (rótulos de qualidade sobre 28.792 fotos do
  EyePACS), cópias redimensionadas ou com filtro gaussiano, e os conjuntos
  "combinados" do Kaggle (APTOS + DDR + IDRiD + EyePACS + Messidor). Repetem
  fotos nossas e, os combinados, também as do Messidor-2.
- **Messidor original (1.200 fotos):** usa uma escala própria, de 0 a 3, e boa
  parte das fotos está dentro do Messidor-2.
- **Só segmentação, ou poucas fotos:** DRIVE, CHASE DB1, HRF, STARE, ROC,
  DiaRetDB0 e 1, E-ophtha, DRiDB, HEI-MED, DR HAGIS, AGAR300.
- **Outra modalidade ou outra escala:** ultra-widefield (a parte UWF do
  DeepDRiD e o UWF-DR de 2026), OCT, JSIEC (39 classes, fora da escala ICDR) e
  ODIR-5K (rótulo "diabetes", sem grau).
- **Coleções recentes sem procedência clara:** por exemplo, as 6.399 fotos de
  cinco estágios no IEEE DataPort e a compilação de 22.454 fotos no Mendeley.

## Cuidados ao juntar

- Descartar as fotos sem grau (a classe "não graduável" do DDR; no DeepDRiD,
  usar a nota de qualidade).
- Rodar `processar_retina` em cada dataset e conferir as máscaras suspeitas: o
  enquadramento muda de câmera para câmera, e as cópias do Kaggle já vêm
  recortadas.
- Os datasets novos entram só no treino, como as fotos extras de 2015 hoje
  (`ampliar_treino`, com a coluna `pasta_processada` e uma coluna de origem).
- Para o app, que é uso fora da pesquisa, conferir a licença de cada um: APTOS
  e EyePACS são dados de competição do Kaggle; BRSET e mBRSET têm o termo de
  uso do PhysioNet; IDRiD é CC BY 4.0.

## Licenças e o uso no app

O projeto é de iniciação científica, e o app seria distribuído de graça para
postos de saúde, sem fins comerciais. Leitura das regras de cada base em
30/10/2026 (não é parecer jurídico):

| Base | Regra | Pesquisa (IC, artigos) | App gratuito nos postos |
|---|---|---|---|
| EyePACS (Kaggle 2015) | "use the Data solely for the purpose and duration of the Competition"; participar não dá licença sobre os dados | Uso acadêmico corrente, mas fora da letra da regra | **Não coberto: pedir autorização à EyePACS** |
| APTOS 2019 (Kaggle) | "non-commercial purposes only, including (…) academic research and education" | Sim | Sim, enquanto for não comercial |
| DDR | CC BY-NC-SA 4.0; uso comercial só com autorização dos autores | Sim, citando o artigo | Sim, sem cobrança; o modelo distribuído deve seguir a mesma licença (BY-NC-SA) |
| IDRiD | CC BY 4.0 | Sim, citando | Sim |
| DeepDRiD | CC BY-SA 4.0 | Sim, citando | Sim; o modelo distribuído deve seguir a mesma licença (BY-SA) |
| Messidor-2 | Grátis para pesquisa e ensino; cópia, redistribuição e uso comercial não autorizado proibidos | Sim (só validação, sem redistribuir as fotos) | Não entra no treino, então não pesa |
| BRSET, mBRSET | Licença de dados credenciados do PhysioNet | Conferir ao ganhar acesso | Conferir ao ganhar acesso |

- **"Não comercial" não depende só de ser grátis:** a Creative Commons olha o
  uso, e a ONG ou o governo podem fazer uso comercial. Sem cobrança, sem
  anúncio e sem contrato pago, a distribuição nos postos é não comercial.
- **O ponto fraco é o EyePACS,** que é a maior parte do treino hoje. Antes de o
  app ir para os postos, vale a orientadora ou a instituição pedir à EyePACS,
  por escrito, autorização para usar os dados da competição de 2015 num
  aplicativo gratuito de saúde pública.
- Licenças "SA" (DDR, DeepDRiD): ao publicar o modelo, usar uma licença
  compatível (CC BY-NC-SA 4.0 cobre as duas) e citar cada base.
- Uso clínico é outro assunto: software que apoia diagnóstico pode ser
  regulado pela ANVISA como dispositivo médico, independentemente dos dados.

## Como baixar direto para o Drive, sem passar pelo PC

Tudo vai para a pasta `Datasets/` da conta pessoal (5 TB), compartilhada só
para leitura com a conta do projeto, uma subpasta por dataset, sem
descompactar.

- **Kaggle** (APTOS, cópias do DDR e do Messidor-2, rótulos do Messidor-2):
  num Colab da conta pessoal, só CPU, com o Drive montado, o comando `kaggle`
  grava direto na pasta do Drive. O token sai em Kaggle → Settings → API →
  Create New Token e vai para os "secrets" do Colab (`KAGGLE_USERNAME` e
  `KAGGLE_KEY`). Por exemplo:
  `kaggle datasets download -d mariaherrerot/ddrdataset -p /content/drive/MyDrive/Datasets/DDR`
- **Google Drive** (DDR original): logado na conta pessoal, abrir o link e
  copiar para o próprio Drive (a cópia acontece no servidor do Google); para
  uma pasta inteira, `gdown --folder <link>` no mesmo Colab.
- **PhysioNet** (BRSET, mBRSET): depois da aprovação do acesso, no Colab da
  conta pessoal,
  `wget -r -N -c -np --user SEU_USUARIO --ask-password https://physionet.org/files/mbrset/1.0/`
  dentro da pasta do Drive (a senha é digitada por você no terminal).
- **IEEE DataPort e ADCIS** (IDRiD, Messidor-2): pedem login ou pedido de
  acesso. São pequenos; baixar pelo navegador e subir, ou usar no Colab o link
  que chegar por e-mail.

O acesso credenciado do PhysioNet e o pedido do Messidor-2 levam dias: vale
começar pelos dois.

O processamento das imagens novas também pode sair do Colab pago: como são
datasets pequenos, um Colab grátis da conta pessoal processa e grava o cache
no Drive pessoal, e os treinos só o restauram. É o desenho original dos
caches, que não funcionou com o EyePACS bruto (~105 GB) por falta de disco.

## Fontes

- DDR: [GitHub nkicsl/DDR-dataset](https://github.com/nkicsl/DDR-dataset), [cópia no Kaggle](https://www.kaggle.com/datasets/mariaherrerot/ddrdataset)
- BRSET: [PhysioNet](https://physionet.org/content/brazilian-ophthalmological/), [artigo (PLOS Digital Health, 2024)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11239107/), [GitHub](https://github.com/luisnakayama/BRSET)
- mBRSET: [PhysioNet](https://www.physionet.org/content/mbrset/), [artigo (Scientific Data, 2025)](https://www.nature.com/articles/s41597-025-04627-3), [GitHub](https://github.com/luisnakayama/mBRSET)
- DeepDRiD: [GitHub deepdrdoc/DeepDRiD](https://github.com/deepdrdoc/DeepDRiD), [GitHub AIMedLab/DeepDRiD](https://github.com/AIMedLab/DeepDRiD)
- IDRiD: [IEEE DataPort](https://ieee-dataport.org/node/956), [grand-challenge](https://idrid.grand-challenge.org/Data/)
- FGADR: [artigo (arXiv 2008.09772)](https://arxiv.org/abs/2008.09772v3)
- Messidor-2: [rótulos do Google no Kaggle](https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades), [cópia recortada no Kaggle](https://www.kaggle.com/datasets/mariaherrerot/messidor2preprocess), [ADCIS](https://www.adcis.net/en/third-party/messidor/)
- EyeQ: [GitHub HzFu/EyeQ](https://www.github.com/HzFu/EyeQ)
- Revisão de 22 bases abertas: [Sensors, 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12473202/)
