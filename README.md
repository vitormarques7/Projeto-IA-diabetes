# Projeto de Inteligência Artificial — Diabetes ML

Template educacional para o desenvolvimento de um projeto acadêmico de Machine Learning na disciplina de Inteligência Artificial. Este repositório fornece apenas uma organização inicial: a definição do problema, a análise dos dados, os modelos e os experimentos deverão ser desenvolvidos pela equipe.

## Equipe

**Nome da equipe:** _preencher_

| Integrante | Conta no GitHub | Responsabilidade inicial |
| --- | --- | --- |
| João Vitor dos Santos Marques | `@vitormarques7` | A definir |
| Ana Oliveira Vanderlei | `@usuario` | A definir |
| Diego Rodrigues | `@DiegoRCJ` | A definir |
| Raquel de Lima Cavalcanti | `@uraqueldelima236-sudo` | A definir |

## Descrição geral

Este projeto consiste na reprodução acadêmica do artigo "Diabetes prediction using machine learning and explainable AI techniques" (Tasin et al., 2023).

O problema abordado é a predição precoce da diabetes mellitus, uma doença crônica que afeta milhões de pessoas e pode causar complicações severas se não diagnosticada a tempo. A relevância do projeto reside em aplicar técnicas de Machine Learning para automatizar esse diagnóstico com alta precisão e, crucialmente, utilizar Explainable AI (LIME e SHAP) para tornar as decisões do algoritmo transparentes para os profissionais de saúde. Nosso objetivo é reproduzir o tratamento de dados relatado no estudo, incluindo a imputação de dados semi-supervisionada para a variável "insulina" e o balanceamento de classes; treinar os classificadores mencionados no artigo e comparar as métricas de avaliação (Acurácia, F1-Score, AUC) obtidas pela equipe com as relatadas pelos autores originais.

## Conjunto de Dados:
O estudo original utiliza uma combinação de dois datasets:
Pima Indians Diabetes Database (disponível no Kaggle/UCI).
RTML Dataset (um dataset privado de pacientes bengalis, disponibilizado publicamente pelos autores no repositório oficial do artigo).

- **Artigo completo e publicado:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10107388/
- **Dataset (Kaggle):** https://www.kaggle.com/datasets/jamaltariqcheema/pima-indians-diabetes-dataset.
</aside>

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
│   └── README.md
├── notebooks/
│   └── .gitkeep
├── src/
│   ├── data/
│   │   └── __init__.py
│   ├── preprocessing/
│   │   └── __init__.py
│   ├── models/
│   │   └── __init__.py
│   ├── evaluation/
│   │   └── __init__.py
│   ├── utils/
│   │   └── __init__.py
│   └── __init__.py
├── tests/
│   ├── .gitkeep
│   └── __init__.py
├── results/
│   ├── figures/
│   │   └── .gitkeep
│   └── metrics/
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
- `.github/`: modelos para Issues e Pull Requests.

Os notebooks devem apoiar a exploração e a comunicação dos experimentos. Coloque código reutilizável em `src/` para evitar concentrar toda a implementação nos notebooks.

## Configuração do ambiente

É recomendado utilizar Python 3.10 ou uma versão mais recente compatível com as dependências do projeto.

1. Clone o repositório e acesse sua pasta:

   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd <NOME_DO_REPOSITORIO>
   ```

2. Crie um ambiente virtual:

   ```bash
   python -m venv .venv
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
   pip install -r requirements.txt
   ```

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

Este template não contém uma solução de Machine Learning. A escolha das técnicas, a implementação, os testes, os experimentos e a análise dos resultados são responsabilidades dos alunos.
