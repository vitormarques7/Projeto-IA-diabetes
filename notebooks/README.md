# Notebooks no VS Code

O notebook `01_preprocessamento.ipynb` usa os mesmos módulos dos comandos de
terminal. Execute antes `python -m src.data.download` na raiz do repositório.
No VS Code, selecione o kernel Python de `.venv` (ou `.venv-validation`, caso
esteja reproduzindo a validação feita durante a implementação).

Não copie a lógica para outros notebooks: importe `load_split()` e
`build_pipeline()` conforme o README. Notebooks podem executar com diretório
inicial na raiz ou nesta pasta; o primeiro bloco localiza o repositório.

As versões antigas estão em `legacy/`, apenas para consulta histórica.
