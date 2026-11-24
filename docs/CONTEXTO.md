# Contexto do projeto (até 24/11/2026)

Resumo de tudo o que foi feito na IC de retinopatia diabética, para entender o
projeto de ponta a ponta e para dar contexto a outra máquina ou outra sessão do
Claude. Os números detalhados de cada rodada estão em `docs/PLANO_V4.md`.

## 1. O que é o projeto

- **Objetivo:** olhar a foto do fundo de olho de uma pessoa com diabetes e dizer
  se há sinal de retinopatia diabética.
- **Uso pretendido:** um app gratuito que o Inatel distribuiria para os postos de
  saúde de Santa Rita. É uma **pré-triagem**, não um diagnóstico. A função é
  lembrar quem precisa procurar o oftalmologista e fazer o acompanhamento.
- **Formato do exame no app:** duas fotos, o olho esquerdo e o direito. O
  resultado é por paciente: se um dos olhos parece doente, o paciente é
  encaminhado.
- **Restrição técnica:** o modelo roda no celular, então tem que ser leve. Por
  isso, uma rede EfficientNet-B0 (pequena), e não a B4 (grande).
- **Repositório:** GitHub `Rcorrea07/ML_Retinopatia`. A branch de trabalho é
  `refatoracao`, mantida igual à `main`.

## 2. Conceitos para ler os números

- **Graus da retinopatia (escala ICDR):** 0 = sem retinopatia, 1 = leve,
  2 = moderada, 3 = grave, 4 = proliferativa. O modelo decide **0 contra 1–4**
  (tem ou não tem). Ele também aprende o grau, como tarefa auxiliar.
- **Sensibilidade:** de cada 100 doentes, quantos o modelo acusa. É a medida
  mais importante numa triagem, porque deixar um doente passar é o pior erro.
- **Especificidade:** de cada 100 saudáveis, quantos o modelo libera. Os
  saudáveis acusados são **falsos positivos**, que geram encaminhamentos à toa.
- **Limiar:** o modelo dá uma nota de 0 a 1, e acima do limiar ele acusa. O
  limiar não é 0,5. Ele é escolhido para pegar **pelo menos 90% dos doentes**
  com o mínimo de falsos positivos. Há um limiar por olho e outro por paciente
  (a nota do paciente é a maior entre os dois olhos).
- **ROC-AUC:** mede se o modelo põe os doentes acima dos saudáveis, sem depender
  do limiar. 1,0 é perfeito e 0,5 é chute. Serve para comparar modelos.
- **Kappa:** concordância entre o grau previsto e o grau real (0 a 1).
- **TTA (aumento no teste):** o modelo vê a foto original e versões espelhadas e
  tira a média. TTA2 = 2 versões, TTA4 = 4. Melhora um pouco sem treinar nada.
- **Validação e teste:** os dados são divididos **por paciente**, para que o
  mesmo paciente nunca apareça em duas partes. O treino ensina. A validação
  escolhe o modelo e o limiar. O teste mede **uma única vez**, no modelo final,
  para o número ser honesto. O teste do EyePACS já foi usado em 03/11 e não se
  abre mais.
- **Validação externa:** medir o modelo em fotos de outro país, câmera e
  população, que ele nunca viu. Mostra se o modelo funciona fora do EyePACS.
- **Pré-processamento de Ben Graham:** antes de treinar, cada foto é recortada
  na retina e tem a cor normalizada. Isso realça lesões e iguala fotos de
  câmeras diferentes. Na campanha de outubro, tirar esse passo derrubou a
  especificidade de 0,671 para 0,609.
- **Unidades do Colab:** o Colab Pro+ (pago pela faculdade) cobra em unidades.
  Uma A100 gasta cerca de 5 a 7 unidades por hora, mesmo parada.

## 3. Dados

| Conjunto | Fotos | Papel |
|---|---:|---|
| EyePACS 2015 (Kaggle), parte `train` | 35.126 | Treino, validação e teste (divididos por paciente) |
| EyePACS 2015, parte `test` (rótulos dos organizadores) | 53.576 | Só treino ("extras") |
| DDR, APTOS, DeepDRiD, IDRiD | 17.631 | Só treino ("externos", desde o R12) |
| Messidor-2 (França) | 1.748 | Só validação externa (V01) |
| mBRSET (Brasil, fotos de celular) | — | Validação externa futura (precisa de acesso no PhysioNet) |

As fotos processadas em 768 px ficam em caches zip no Drive pessoal (pasta
`ProjetoRetinopatia_caches`, ~19,5 GB), compartilhada só para leitura com a
conta do projeto. Assim, um treino não gasta horas de A100 reprocessando
imagens.

**Licenças:** o projeto é de pesquisa e o app seria gratuito, então os datasets
externos permitem o uso. A exceção é o **EyePACS do Kaggle**, cuja licença vale
só para a competição. Para usar o app nos postos, falta uma autorização por
escrito da EyePACS, que a orientadora pode pedir.

## 4. Linha do tempo

1. **Treino v2 (início de outubro):** B4 448 px. No teste, por paciente:
   ROC 0,915 e especificidade 0,70. Levou ~10 h porque o aumento de dados rodava
   na CPU.
2. **v3 e dados extras (06 a 10/10):** pré-processamento corrigido (máscara da
   retina melhor e Ben só dentro da retina), aumento de dados movido para a GPU
   e as 53 mil fotos extras no treino. Com B4 448: ROC 0,923 por paciente no
   teste.
3. **Campanha de outubro (E01–E11, 20/10):** ajustes finos na B0 448. Quase
   tudo empatou no ruído. O que importou foi a normalização de Ben, a resolução
   e o TTA. Referência E01, por paciente na validação: ROC 0,908 e
   especificidade 0,671.
4. **Plano v4 (22/10 a 03/11):** rodadas R01–R12, comparadas sempre na
   validação.
   - Resolução maior ajudou: B0 640 (R02) e B0 768 (R10).
   - Destilação B4 → B0 (R03) piorou e foi descartada.
   - EfficientNetV2-B0 empatou, mas é mais pesada no celular.
   - Treino mais longo (R06) não ajudou.
   - Cada configuração rodou com duas sementes, para separar ganho real de
     sorte.
   - **R12 (B0 768 + datasets externos)** foi o melhor: especificidade por
     paciente com TTA2 de 0,801 (semente 42).
   - A partir da rodada 4, vários treinos passaram a rodar juntos numa sessão
     só, que desliga sozinha no fim, para gastar menos unidades.
5. **R09, o modelo final no teste (03/11):** R12 semente 42 com TTA2, medido uma
   vez no teste do EyePACS. Depois, exportado para TFLite.
6. **Refatoração (novembro, branch `refatoracao`):** o código feito rodada a
   rodada foi reorganizado.
   - Regras de medição num módulo só (`metricas.py`).
   - Uma ficha por fonte de dados (`fontes.py`).
   - Um pipeline de treino só (`execucao.py`).
   - Notebooks com formulário em vez de cópias por execução.
   - Testes e uma conferência que recalcula as métricas antigas número a
     número.

   Os números antigos foram reproduzidos.
7. **V01, validação externa no Messidor-2 (19/11):** ver a seção 6.
8. **F10, RETFound-Green (24/11):** um modelo pré-treinado em fotos de retina
   (ViT pequeno, PyTorch), ajustado nos mesmos dados do R12 e comparado na
   validação. **Perdeu:** especificidade por paciente com TTA2 de 0,660, contra
   0,801 do R12. Também é proibido no app, porque a licença é de pesquisa não
   comercial e barra projetos com empresa. Ficou só como comparação.

## 5. O modelo final

**R12 semente 42:** EfficientNet-B0, fotos de 768 px, treinado com EyePACS,
extras e externos, e TTA2.

Teste do EyePACS (03/11):

| Medida | Por paciente | Por olho |
|---|---:|---:|
| ROC-AUC | 0,934 [0,920; 0,947] | 0,939 |
| Sensibilidade | 0,894 | 0,909 |
| Especificidade | 0,795 [0,772; 0,817] | 0,783 |

- **Por grau:** o grau 1 (leve) é o mais difícil, com 76,5% detectados. Os
  graus graves passam de 94%. Um em cada cinco olhos saudáveis (21,7%) é
  marcado.
- **Retinopatia referenciável (grau ≥ 2):** ROC 0,958 por paciente.
- **Arquivo do app:** TFLite fp16 com 8,1 MB, que dá os mesmos resultados do
  modelo original. A versão int8 (4,6 MB) foi descartada porque mudava as
  notas.

## 6. Messidor-2: o ponto que precisa de conversa

Os números são por paciente, com os dois olhos graduáveis (870 pacientes):

| Medida | Messidor-2 | Validação do EyePACS |
|---|---:|---:|
| ROC-AUC | 0,928 [0,909; 0,944] | 0,930 |
| Sensibilidade | 0,948 [0,926; 0,967] | ~0,90 |
| Especificidade | **0,559** [0,516; 0,604] | 0,801 |

**Leitura.** O ROC quase não mudou. O modelo continua separando bem doentes de
saudáveis numa câmera e população que nunca viu. O que mudou foi a escala das
notas: no Messidor-2, o modelo dá notas mais altas para todo mundo. Com o mesmo
limiar do EyePACS, mais saudáveis passam da linha. Por isso, 44% dos saudáveis
seriam encaminhados, contra 20% no EyePACS. A sensibilidade subiu para 0,948.
Para retinopatia referenciável (grau ≥ 2), o resultado é bom: ROC 0,976,
sensibilidade 0,970 e especificidade 0,771.

**Por que não ajustar o limiar no Messidor-2.** O Messidor-2 é a prova externa.
Escolher o limiar olhando para ele seria ajustar o modelo na própria prova, e o
número deixaria de ser honesto.

**Caminhos para discutir com o orientador:**
1. Aceitar como está. Na pré-triagem, um falso positivo vira "procure o
   oftalmologista", mas sobrecarrega os encaminhamentos.
2. Calibrar por câmera. Juntar algumas centenas de fotos com laudo da câmera
   que os postos vão usar, ajustar o limiar nelas e medir em outra parte. É o
   caminho mais realista para o app.
3. Usar o mBRSET quando o acesso sair: uma parte ajusta o limiar e a outra
   mede. Isso também mostra se o deslocamento acontece com fotos brasileiras.

## 7. Onde está cada coisa

- **Código:** `src/retinopatia/`, com a lógica toda em português. Os notebooks
  em `notebooks/` só chamam funções do pacote.
- **Execução:** tudo roda no Google Colab, porque precisa de GPU. Na máquina
  local não há TensorFlow, e nada deve ser instalado. Uma mudança só chega ao
  Colab depois do `git push`, porque o notebook clona o repositório.
- **Notebooks:**
  - `Retinopatia_0_1` (treino);
  - `Treinos_Paralelos` (vários treinos numa sessão);
  - `Modelo_Final` (R09);
  - `Validacao_Externa` (Messidor-2);
  - `RETFound_Green`;
  - `Analisar_Resultados` (só CPU);
  - `Preparar_Caches`;
  - `Baixar_Datasets`.
- **Drive da conta do projeto, pasta `MyDrive/ProjetoRetinopatia/`:**
  - `modelos/`: os `.keras` e `tflite/`;
  - `resultados/`: uma pasta por execução, com predições, histórico e
    configuração;
  - `relatorios/`;
  - `dados_internos/`: splits e caches de 448.
- **Contas Google:**
  - **conta do projeto:** roda o Colab Pro+ e precisa de 2FA no celular da
    orientadora;
  - **conta pessoal:** guarda os datasets e os caches.

  No Brave, a conta do projeto é `authuser=1`. O mesmo Drive tem o projeto de
  hanseníase (HansenIA), que **não deve ser tocado**.
- **Documentos:**
  - `docs/PLANO_V4.md`: todos os resultados;
  - `docs/DATASETS.md`: datasets e licenças;
  - `docs/REFERENCIAS.md`: soluções do Kaggle que guiaram as escolhas;
  - `CLAUDE.md`: regras do código para o Claude.

## 8. Regras combinadas

- O teste do EyePACS não se abre mais. Modelos novos são comparados na
  validação e na validação externa.
- Messidor-2 e mBRSET nunca entram no treino.
- Nada do RETFound-Green vai para o app, nem como professor de destilação, sem
  permissão escrita do autor.
- Antes de cada rodada no Colab, ler o saldo de unidades em Ambiente de
  execução → Ver recursos, e de novo no fim. Saldo em 24/11: 884,8 unidades.
- Commits: normalmente o Claude prepara os comandos `git add` e `git commit` e
  o usuário cola. Push nas duas branches: `git push && git push origin
  refatoracao:main`.
- Sem assinar armazenamento extra no Drive.
- Senhas, tokens e 2FA ficam sempre com o usuário ou a orientadora.

## 9. O que falta

**Do modelo:**
1. Conversar com o orientador sobre a especificidade no Messidor-2 (seção 6).
2. mBRSET: fazer o curso CITI ("Data or Specimens Only Research"), enviar o
   relatório de conclusão no PhysioNet, pedir o credenciamento (referência = o
   orientador) e assinar o termo de uso. Depois, medir o R12 nele com o mesmo
   código do Messidor-2.
3. Opcional: RETFound-Green com fotos sem Ben, como no pré-treino dele. Custa
   um cache novo (~2 h) e uma rodada, e mesmo assim não poderia ir para o app.

**Do produto:**
1. App com 2 fotos por exame (com o outro bolsista).
2. Autorização da EyePACS para usar no app um modelo treinado com os dados
   dela.

**Tarefas do usuário:**
- subir os PDFs dos relatórios de 03/11 e 18/11 no OneDrive;
- preparar a apresentação da 1ª Mostra de Pesquisa;
- pesquisar ferramentas aceitas em saúde (banco de dados, segurança, APIs).
