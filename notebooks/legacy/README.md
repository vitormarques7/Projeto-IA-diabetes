# Versões anteriores

Estes notebooks preservam o trabalho anterior da equipe e não são o fluxo ativo.
Podem conter caminhos relativos antigos e chamadas específicas do Google Colab.
Não use seus CSVs para comparar resultados com o protocolo Pima + RTML atual.

- `Pre_Processamento.ipynb`: versão que ajustava médias e escala antes da divisão.
- `Tratamento_de_dados.ipynb`: avanço que separava treino/teste antes de ajustar
  médias e escala, preservando Outcome; ainda usava apenas o Pima.
- `codigo_normalizacao.ipynb`: tratamento sem escala preservado da branch
  `issue-1-estrutura`; o nome original era `codigo_normalização.ipynb`.

O fluxo suportado está no README principal e no notebook local de preparação.
Os CSVs antigos foram preservados localmente em `data/processed/legacy/`;
também continuam recuperáveis no histórico do Git. Dados não são versionados
na implementação atual.
