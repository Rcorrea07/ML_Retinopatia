"""
Pacote do projeto de detecção de retinopatia diabética.

A execução acontece no Google Colab, através do notebook
`notebooks/Retinopatia_0_1.ipynb`, que apenas orquestra as funções
definidas aqui.

Módulos:
    configuracao      -> constantes, caminhos e hiperparâmetros
    utilitarios       -> sementes, determinismo, precisão mista e
                         registro da configuração do experimento
    ambiente_colab    -> Drive, extração do dataset, cache das imagens
                         processadas e download
    conjunto_dados    -> leitura do CSV, rótulos e divisão treino/
                         validação/teste por paciente
    pre_processamento -> máscara da retina, recorte, cor de Ben Graham
                         e máscara circular
    pipeline_dados    -> pipelines tf.data e bloco de data augmentation
    modelo            -> EfficientNet + GeM + cabeças binária e de grau
    treinamento       -> pesos de classe, callbacks e as duas fases
    avaliacao         -> curvas, TTA, limiar e avaliação por olho e paciente
    diagnostico       -> erros por grau, alvo referenciável, cabeça de
                         grau, agregação por paciente e bootstrap
"""
