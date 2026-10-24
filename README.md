# Projeto de Inteligência Artificial — Diabetes ML

Projeto acadêmico de reprodução de Machine Learning na disciplina de Inteligência Artificial, organizado conforme o template do professor. O pré-processamento Pima + RTML é compartilhado; os classificadores e a análise dos experimentos continuam sob responsabilidade dos integrantes.

## Equipe

**Nome da equipe:** _preencher_

| Integrante | Conta no GitHub | Responsabilidade inicial |
| --- | --- | --- |
| João Vitor dos Santos Marques | `@vitormarques7` | A definir |
| Ana Oliveira Vanderlei | `@usuario` | Implementar modelo de Árvore de Decisão |
| Diego Rodrigues | `@DiegoRCJ` | Tratamento de dados e MLP |
| Raquel de Lima Cavalcanti | `@raqueldelima236-sudo` | implementar regressão logística |

## Descrição geral

Este projeto consiste na reprodução acadêmica do artigo "Diabetes prediction using machine learning and explainable AI techniques" (Tasin et al., 2023).

O problema abordado é a predição precoce da diabetes mellitus, uma doença crônica que afeta milhões de pessoas e pode causar complicações severas se não diagnosticada a tempo. A relevância do projeto reside em aplicar técnicas de Machine Learning para automatizar esse diagnóstico com alta precisão e, crucialmente, utilizar Explainable AI (LIME e SHAP) para tornar as decisões do algoritmo transparentes para os profissionais de saúde. Nosso objetivo é reproduzir o tratamento de dados relatado no estudo, incluindo a imputação de dados semi-supervisionada para a variável "insulina" e o balanceamento de classes; treinar os classificadores mencionados no artigo e comparar as métricas de avaliação (Acurácia, F1-Score, AUC) obtidas pela equipe com as relatadas pelos autores originais.

## Conjunto de Dados:
O estudo original utiliza uma combinação de dois datasets:
Pima Indians Diabetes Database (disponível no Kaggle/UCI).
RTML Dataset (um dataset privado de pacientes bengalis, disponibilizado publicamente pelos autores no repositório oficial do artigo).

- **Artigo completo e publicado:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10107388/
- **Github do estudo original:** https://github.com/tansin-nabil/Diabetes-Prediction-Using-Machine-Learning
- **Dataset (Kaggle):** https://www.kaggle.com/datasets/jamaltariqcheema/pima-indians-diabetes-dataset.

A reprodução utiliza o Pima bruto (768 pacientes) e a versão RTML publicada com
SkinThickness (109 pacientes), totalizando 877. A diferença em relação aos 203
participantes descritos no artigo está documentada em [data/README.md](data/README.md),
junto às correções de vazamento e às fontes fixadas em [data/sources.json](data/sources.json).

## Estrutura do repositório

```text
.
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   └── task.md
│   └── pull_request_template.md
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   ├── processed/
│   │   └── .gitkeep
│   ├── README.md
│   └── sources.json
├── notebooks/
│   ├── 01_preprocessamento.ipynb
│   ├── README.md
│   └── legacy/                    # versões anteriores preservadas
├── src/
│   ├── data/
│   │   ├── download.py
│   │   ├── loader.py
│   │   ├── schema.py
│   │   └── __init__.py
│   ├── preprocessing/
│   │   ├── pipeline.py
│   │   ├── diagnostics.py
│   │   ├── prepare.py
│   │   └── __init__.py
│   ├── models/
│   │   └── __init__.py
│   ├── evaluation/
│   │   └── __init__.py
│   ├── utils/
│   │   ├── paths.py
│   │   └── __init__.py
│   └── __init__.py
├── tests/
│   ├── conftest.py
│   ├── test_preprocessing.py
│   └── __init__.py
├── results/
│   ├── figures/
│   │   └── .gitkeep
│   ├── metrics/
│   │   └── .gitkeep
│   └── artifacts/
│       └── .gitkeep
├── .gitignore
├── README.md
└── requirements.txt
```

### Finalidade dos diretórios

- `data/raw/`: datasets originais, sem alterações. Os arquivos de dados não devem ser versionados por padrão.
- `data/processed/`: dados gerados por limpeza, transformação ou preparação. Esses arquivos também não devem ser versionados por padrão.
- `notebooks/`: análises exploratórias, experimentos e registros de investigação em Jupyter.
- `src/data/`: código reutilizável de carregamento e manipulação inicial dos dados.
- `src/preprocessing/`: código reutilizável de pré-processamento.
- `src/models/`: código de modelos e treinamento a ser desenvolvido pelos alunos.
- `src/evaluation/`: código de avaliação e métricas a ser desenvolvido pelos alunos.
- `src/utils/`: funções auxiliares compartilhadas pelo projeto.
- `tests/`: testes automatizados do código produzido pela equipe.
- `results/figures/`: figuras e gráficos gerados pelos experimentos.
- `results/metrics/`: resultados de métricas gerados pelos experimentos.
- `results/artifacts/`: transformadores ajustados, ignorados pelo Git.
- `.github/`: modelos para Issues e Pull Requests.

Os notebooks devem apoiar a exploração e a comunicação dos experimentos. Coloque código reutilizável em `src/` para evitar concentrar toda a implementação nos notebooks.

## Configuração do ambiente

Use **Python 3.11**, versão usada na validação. As dependências diretas estão
fixadas em `requirements.txt`; as versões efetivamente usadas são registradas
no relatório de preparação. No macOS, o XGBoost requer o runtime OpenMP:

```bash
brew install libomp
```

1. Clone o repositório e acesse sua pasta:

   ```bash
   git clone https://github.com/vitormarques7/Projeto-IA-diabetes.git
   cd Projeto-IA-diabetes
   ```

2. Crie um ambiente virtual:

   ```bash
   python3.11 -m venv .venv
   ```

3. Ative o ambiente virtual:

   No Linux ou macOS:

   ```bash
   source .venv/bin/activate
   ```

   No Windows (PowerShell):

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

4. Instale as dependências:

   ```bash
   python -m pip install -r requirements.txt
   ```

No Windows, crie o ambiente com `py -3.11 -m venv .venv`. Se houver bloqueio
de ativação no PowerShell, use diretamente `.venv\Scripts\python.exe`.

## Execução no VS Code

Abra a pasta do repositório no VS Code. Instale as extensões **Python** e,
para notebooks, **Jupyter**. Em **Python: Select Interpreter**, selecione
`.venv/bin/python` (Windows: `.venv\Scripts\python.exe`). No notebook, selecione
também esse ambiente em **Select Kernel**. O fluxo principal usa o terminal
integrado e não depende do Google Colab.

Na raiz do projeto, com o ambiente ativo:

```bash
python -m src.data.download
python -m src.preprocessing.prepare
python -m pytest -q
```

O download é reutilizável, verifica SHA-256 e não altera arquivos já existentes.
A preparação gera os cenários `none`, `smote` e `adasyn`, cada um com
`train.csv` e `test.csv` em `data/processed/`. As tabelas contêm sete
características, nesta ordem: `Pregnancies`, `Glucose`, `BloodPressure`,
`SkinThickness`, `Insulin`, `BMI`, `Age`; `Outcome` é a resposta separada.

O relatório `results/metrics/preprocessing.json` registra configurações,
versões, contagens antes/depois, médias, parâmetros de escala, pacientes usados
na regressão e RMSE de XGB/SVR/GPR. A informação mútua fica em CSV e figura;
as pipelines ajustadas ficam em `results/artifacts/`. Tudo é gerado localmente.
O notebook `notebooks/01_preprocessamento.ipynb` oferece uma visualização local
do mesmo fluxo. Os notebooks anteriores estão em `notebooks/legacy/`.

## Integração com o modelo de cada integrante

Todos devem usar a mesma divisão 80/20 estratificada, semente 0: 701 pacientes
de treino e 176 de teste. `load_split()` retorna dados ainda não transformados,
com identificadores no índice, origem em `_source` e metadados da divisão.
`build_pipeline()` remove origem e pedigree antes do classificador e impede
que o rótulo seja usado para prever insulina.

Exemplo de integração com um estimador scikit-learn (substitua-o pelo seu):

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
from src.data import load_split
from src.preprocessing import build_pipeline

split = load_split()
pipeline = build_pipeline(LogisticRegression(max_iter=1000), sampling="adasyn")
search = GridSearchCV(
    pipeline,
    {"model__C": [0.1, 1.0, 10.0]},
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
    scoring="roc_auc",
    error_score="raise",
)
search.fit(split.X_train, split.y_train)
# Avalie uma vez após escolher os parâmetros apenas no treino.
probabilities = search.predict_proba(split.X_test)[:, 1]
print(roc_auc_score(split.y_test, probabilities))
```

Use `sampling="none"`, `"smote"` ou `"adasyn"`. As etapas de imputação,
regressão, reamostragem e escala são ajustadas novamente em cada fold.
O regressor de insulina usa somente Pima de treino com respostas observadas;
`Outcome` nunca entra como preditor. O diagnóstico de informação mútua não
seleciona características usando o teste.

Os CSVs já transformados servem para treinamento direto com parâmetros
predefinidos. **Não faça GridSearchCV ou validação cruzada sobre esses CSVs**:
isso reutilizaria transformações aprendidas fora dos folds. Para previsões com
uma pipeline completa, forneça os dados brutos de `load_split()`, incluindo
`_source`; não forneça dados que já foram normalizados. Para novas pessoas,
será necessário definir um contrato de entrada específico em outra atividade.

Após esta correção, cada integrante deve atualizar sua branch com a `main`
após o merge, adaptar o carregamento e refazer treinamento e métricas. Não
compare diretamente métricas dos antigos CSVs Pima com o novo Pima + RTML.
Esta entrega não implementa os classificadores individuais, SHAP ou LIME.

## Verificação

Os testes sintéticos não precisam de rede. Após o download, também são
executados os testes de integração com os 877 pacientes. Sem as fontes locais,
esses testes de integração são explicitamente marcados como ignorados.
São verificados: disjunção dos pacientes, dados brutos preservados, regras de
zeros, sete características finais, independência dos rótulos na imputação,
ausência de ajuste no teste, balanceamento somente no treino, serialização,
reprodução e reajuste em cada fold. Igualdade com as métricas do artigo não
é critério de aprovação.

## Organização dos dados

Armazene os datasets originais em `data/raw/` e preserve-os sem modificações. Salve em `data/processed/` apenas os dados resultantes de limpeza, transformação ou preparação.

Datasets e artefatos gerados podem ser grandes ou conter informações que não devem ser publicadas. Por isso, o `.gitignore` impede o versionamento desses arquivos por padrão. Os arquivos `.gitkeep` mantêm a estrutura das pastas no Git. Se a equipe precisar compartilhar dados, deve combinar um meio apropriado e documentar como obtê-los em `data/README.md`.

## Fluxo de desenvolvimento

O fluxo esperado é:

```text
Issue → Branch → Commits → Pull Request → Code Review → Approval → Merge
```

Regras do projeto:

1. Toda alteração de código deve estar associada a uma Issue.
2. Cada atividade deve ser desenvolvida em uma branch própria.
3. Não devem ser realizados commits diretamente na `main`.
4. Toda alteração deve entrar na `main` por meio de Pull Request.
5. O Pull Request deve estar associado à Issue correspondente. Quando o merge concluir a atividade, use `Closes #numero_da_issue` na descrição do PR.
6. O autor do Pull Request não pode aprovar o próprio PR.
7. Todo PR precisa de pelo menos uma aprovação de outro integrante da equipe antes do merge.
8. Cada integrante deve utilizar sua própria conta GitHub e realizar seus próprios commits.
9. O histórico do GitHub será utilizado para acompanhar a participação individual dos integrantes.

### Exemplo de início de uma atividade

Depois de criar ou escolher uma Issue, atualize a `main` local e crie uma branch com um nome descritivo:

```bash
git switch main
git pull
git switch -c issue-12-descricao-curta
```

Faça commits pequenos e claros. Ao concluir a atividade, envie a branch e abra um Pull Request utilizando o modelo do repositório.

## Observação

O repositório partiu do template da disciplina. O pré-processamento agora é
compartilhado; a implementação dos classificadores, os experimentos e a análise
dos resultados permanecem como responsabilidades dos integrantes.
