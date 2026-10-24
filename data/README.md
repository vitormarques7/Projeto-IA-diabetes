# Dados do projeto

Na raiz do repositório, execute `python -m src.data.download`. O comando usa
URLs fixadas em commits e valida os checksums SHA-256 de `sources.json` antes
de salvar. Uma cópia local com checksum diferente é rejeitada, sem sobrescrita.
`python -m src.preprocessing.prepare` valida novamente os bytes antes da leitura.

## Fontes utilizadas

| Fonte | Arquivo local em raw/ | Registros | Negativos / positivos |
| --- | --- | --- | --- |
| Pima, cópia bruta sem cabeçalho | pima.csv | 768 | 500 / 268 |
| RTML dos autores, sem insulina | rtml_without_insulin.xlsx | 109 | 86 / 23 |
| União na ordem Pima, depois RTML | gerada em memória | 877 | 586 / 291 |

- [Pima: arquivo público usado como fonte](https://github.com/jbrownlee/Datasets/blob/d20fcb6402ae34e653d4513b00f39257bb37ed7f/pima-indians-diabetes.data.csv).
- [RTML: arquivo dos autores](https://github.com/tansin-nabil/Diabetes-Prediction-Using-Machine-Learning/blob/23b982a20815c4965da47b4b1745c5d3ef0735d1/RTML%20without%20Insulin.xlsx).
- Os commits, URLs e SHA-256 completos estão no manifesto `sources.json`.

O artigo descreve 203 participantes, mas o arquivo RTML com SkinThickness
publicado e utilizado pelos notebooks tem 109. Isso explica os 877 registros
observados no código. O arquivo `Raw_Data_Without_Insulin.xlsx`, com 203, não
possui SkinThickness e não é a fonte deste experimento. Não inventamos essa
coluna nem removemos pacientes para reproduzir contagens. A contagem de classes
do artigo (669/302) também diverge dos arquivos publicados (586/291).

O CSV `RTML with Insulin.csv` dos autores é apenas referência externa: a insulina
é recalculada localmente, porque a versão publicada foi gerada com outro ajuste.
Arquivos antigos `diabetes.csv` e `Raw_Data_Without_Insulin.xlsx` que já existam
no computador são preservados, mas não são lidos pelo fluxo atual. O antigo
`diabetes.csv` local já estava tratado, portanto não foi reutilizado como bruto.

Uso: fontes disponibilizadas publicamente para pesquisa; esta reprodução faz
download direto da origem e não redistribui os registros. Consulte os termos
das fontes originais antes de qualquer outra utilização; disponibilidade no
GitHub não equivale a uma licença adicional de redistribuição.

## Saídas locais

- `processed/raw_train.csv` e `raw_test.csv`: divisão sem imputação nem escala,
  incluindo IDs e origem; `load_split()` regenera a mesma divisão dos brutos.
- `processed/split.csv`: patient_id, source e partition, para auditar a divisão.
- `processed/{none,smote,adasyn}/{train,test}.csv`: sete características finais
  na ordem documentada e Outcome; sem índice, origem ou pedigree nas entradas.
- `processed/legacy/`: cópias locais dos CSVs antigos, fora do fluxo ativo.

Os diretórios de dados são ignorados pelo Git. Os arquivos brutos permanecem
sem alterações após download. As saídas podem ser regeneradas pelos comandos
do README. Os IDs identificam linhas da versão fixada, não pessoas reais.

## Protocolo e divergências documentadas

A divisão estratificada usa semente 0: treino 701 (468/233), teste 176 (118/58).
Somente os pacientes de treino participam dos ajustes de imputação, regressão,
informação mútua e escala. O teste passa apenas pelas transformações aprendidas.

Zeros viram ausentes em Glucose, BloodPressure, SkinThickness, Insulin e BMI,
como no notebook dos autores. Pregnancies=0 e Outcome=0 são preservados.
Médias dos seis preditores são aprendidas no treino recebido pela pipeline.
As insulinas Pima ausentes usam a média das insulinas Pima realmente observadas
nesse treino. As insulinas RTML são previstas por XGB ajustado somente nos
pacientes Pima de treino com insulina observada, sem usar Outcome ou pedigree.

Na comparação interna de regressão (80/20, semente 0), cada modelo recebe uma
imputação ajustada somente no treino interno. Entradas permanecem em unidades
originais; SVR e GPR usam os defaults das versões fixadas (GPR com semente 0).
XGB usa profundidade 10, learning_rate=0.1, 100 estimadores e objetivo
reg:squarederror. O relatório registra o vencedor local, mesmo que não seja XGB.
Por fidelidade ao método escolhido no artigo, XGB continua o regressor usado.
Não são usados valores imputados como respostas reais da regressão.

A informação mútua é um diagnóstico das oito características do Pima de treino,
após imputação pelo próprio treino. A exclusão de DiabetesPedigreeFunction é
fixada pelo protocolo do artigo, não por uma seleção baseada no teste local.
As características finais são Pregnancies, Glucose, BloodPressure, SkinThickness,
Insulin, BMI e Age. Nenhum valor extremo observado é removido automaticamente.

A ordem limpeza/insulina -> SMOTE ou ADASYN -> Min-Max mantém o código dos
autores, inclusive as distâncias de reamostragem nas unidades originais.
SMOTE/ADASYN usam cinco vizinhos, sampling_strategy='minority' e semente 0.
ADASYN pode gerar contagens próximas, mas não exatamente iguais. A escala usa
somente o treino de cada cenário; valores de teste fora de [0, 1] são válidos.

Correções em relação ao código original: separar antes de aprender médias;
excluir Outcome da regressão de insulina; excluir o teste externo de todos os
ajustes; treinar o regressor somente com respostas observadas; reajustar as
etapas dentro de cada fold de validação cruzada. Essas diferenças impedem
tratar as métricas locais como uma réplica numérica exata dos autores.

Os CSVs prontos servem ao treinamento direto de um classificador. Para busca
de hiperparâmetros e validação cruzada, use os dados não transformados de
`load_split()` e `build_pipeline()`, evitando vazamento entre folds.

## Validação local de referência

Em Python 3.11.13/macOS ARM, com as dependências fixadas, o treino original tem
468 negativos/233 positivos; SMOTE gera 468/468 e ADASYN, 468/420. Todos os
cenários preservam os mesmos 176 pacientes de teste (118/58).

Na comparação interna de insulina, os RMSE foram XGB 87,34; SVR 82,48; GPR
169,11. XGB foi mantido por protocolo, e a divergência é registrada no JSON.
Esses resultados avaliam a regressão de insulina, não a classificação de
diabetes. Consulte o relatório gerado para os números completos da sua execução.
