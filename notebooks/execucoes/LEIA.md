# Execuções

Cópias de `notebooks/Retinopatia_0_1.ipynb` que diferem só na primeira célula
(revisão do código, experimento, fumaça e semente), para abrir cada execução
direto do GitHub numa sessão própria do Colab, sem editar células. A lógica
continua no pacote; se o notebook de treino mudar, gere as cópias de novo.

R07 e R08 repetem o R02 (B0 640) e o R05 (V2-B0 640) com
`RETINOPATIA_SEED_TREINO = "17"`: mesmos pacientes em cada conjunto, outra
semente de treino. As pastas de resultado levam `seed17` no nome. As cópias de
R01 a R06 são anteriores à linha da semente e usam a padrão (42).

`Rodada4_768.ipynb` é uma cópia de `notebooks/Treinos_Paralelos.ipynb`: uma
sessão só roda o R10 com a semente 17 e o R11 (V2-B0 768) com as sementes 42
e 17, ao mesmo tempo na mesma A100. As imagens são processadas uma vez, cada
treino passa antes por uma fumaça de poucos minutos, e a sessão desliga
sozinha no fim (também se der erro).

`R09_final_b0_768.ipynb` é uma cópia de `notebooks/Modelo_Final.ipynb`: mede o
B0 768 escolhido no teste, uma vez só, e gera os arquivos TFLite. Abre o teste
(`RETINOPATIA_AVALIAR_TESTE = "1"`); não rode de novo depois de olhar o
resultado.
