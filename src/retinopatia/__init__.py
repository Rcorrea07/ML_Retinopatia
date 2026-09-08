"""
Pacote do projeto de detecção de retinopatia diabética.

A execução acontece no Google Colab, através do notebook
`notebooks/Retinopatia_0_1.ipynb`, que apenas orquestra as funções
definidas aqui.

Módulos:
    configuracao      -> constantes, caminhos e hiperparâmetros
    utilitarios       -> sementes e determinismo
    ambiente_colab    -> montagem do Drive, extração do dataset e download
    conjunto_dados    -> leitura do CSV, rótulos e divisão treino/validação
    pre_processamento -> recorte da retina com o método de Otsu
    pipeline_dados    -> geradores de imagens do Keras
    modelo            -> arquitetura EfficientNetB0 e métricas
    treinamento       -> pesos de classe, callbacks e as duas fases de treino
    avaliacao         -> curvas, limiar de decisão e matriz de confusão
"""
