Read-Me do Curso de Python para Finanças: Análise de Dados e Machine Learning

# Python para Finanças: Análise de Dados e Machine Learning

Este repositório contém os scripts em Python utilizados no curso de Python para Finanças, ministrado pelo Professor Jones Granatyr da IA Expert Academy. O curso aborda conceitos de análise de dados financeiros e machine learning aplicados ao mercado de ações.

## Aulas de Análise de Dados

### 1. Importação e Visualização de Dados
Neste script, utilizamos as bibliotecas pandas, numpy e matplotlib para importar e visualizar os dados de ações salvos em arquivo .csv, destacando a evolução dos preços ao longo do tempo.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importando o conjunto de dados
dataset = pd.read_csv('acoes_02.csv')

# Visualizando as primeiras linhas do conjunto de dados
print(dataset.head())
```

### 2. Cálculo de Taxa de Retorno Simples
Neste script, realizamos o cálculo da taxa de retorno simples para comparar o desempenho das ações ao longo do tempo.

```python
# Efetuando a exclusão da coluna 'Date'
dataset.drop(labels=['Date'], axis=1, inplace=True)

# Calculando as taxas de retorno
taxas_retorno = (dataset / dataset.shift(1)) - 1
print(taxas_retorno.head())
```

### 3. Cálculo do Desvio Padrão e Anualização
Calculamos o desvio padrão das taxas de retorno e realizamos a anualização para avaliar a volatilidade das ações.

```python
# Calculando o desvio padrão em percentuais
desvio_padrao_percentual = taxas_retorno.std() * 100
print(desvio_padrao_percentual)

# Anualizando o desvio padrão
dias_ano = 246
desvio_padrao_anualizado = desvio_padrao_percentual * math.sqrt(dias_ano)
print(desvio_padrao_anualizado)
```

## Aulas de Construção de Carteira

### 4. Construção de Carteira de Ações
Neste script, importamos dados de ações, normalizamos os preços e criamos uma carteira ponderada.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Lista de siglas das ações
acoes = ['ABEV3.SA', 'ODPV3.SA', 'VIVT3.SA', 'PETR4.SA', 'BBAS3.SA', 'BOVA11.SA']

# Criando DataFrame com preços das ações
acoes_df = pd.DataFrame()
for acao in acoes:
    acoes_df[acao] = yf.download(acao, start='2015-01-01', end='2020-11-04')['Close']

# Normalizando os preços
acoes_normalizadas = acoes_df.copy()
for i in acoes_normalizadas.columns[1:]:
    acoes_normalizadas[i] = acoes_normalizadas[i] / acoes_normalizadas[i][0]

# Criando arquivo CSV
acoes_normalizadas.to_csv('acoes.csv')
```

### 5. Análise Descritiva da Carteira
Realizamos uma análise descritiva das ações da carteira, calculando estatísticas como média, desvio padrão e percentual de retorno anual.

```python
# Lendo arquivo CSV
acoes_df = pd.read_csv('acoes.csv')

# Descrevendo as ações
descricao_acoes = acoes_df.describe()
print(descricao_acoes)
```

### 6. Retorno da Carteira e Comparação com o Ibovespa
Calculamos o retorno anual da carteira de ações e comparamos com o retorno do Ibovespa.

```python
# Calculando o retorno anual da carteira
retorno_anual = retorno_carteira.mean() * 246
print(retorno_anual)

# Comparando com o retorno do Ibovespa
retorno_bova = retorno_anual['BOVA']
print(f"Retorno do Ibovespa: {retorno_bova}")
```

## Aulas de Estatística Financeira

### 7. Cálculo da Variância e Volatilidade
Neste script, realizamos o cálculo manual da variância e utilizamos o numpy para calcular a variância e volatilidade das taxas de retorno de uma ação.

```python
# Taxas de retorno de uma ação
taxas_cvc = np.array([-11.86, 63.73, 74.52, 20.42

, -26.39, -5.99, -6.76, 7.65, -12.04, 17.85])

# Cálculo da média e variância
media_cvc = np.mean(taxas_cvc)
variancia_cvc = np.var(taxas_cvc)

# Cálculo da volatilidade
volatilidade_cvc = np.sqrt(variancia_cvc)

print(f"Média: {media_cvc}, Variância: {variancia_cvc}, Volatilidade: {volatilidade_cvc}")
```

### 8. Correlação e Covariância
Calculamos a matriz de covariância e a correlação entre duas ações para entender a relação entre seus retornos.

```python
# Taxas de retorno de duas ações
taxas_ambev = np.array([2.55, 7.21, 3.91, -1.82, -4.17, 2.09, 5.29, -3.33, -0.60, 1.50])
taxas_itau = np.array([-3.48, 8.36, 6.71, 1.02, -1.45, 3.52, 0.21, -2.15, 4.04, -0.56])

# Cálculo da covariância e correlação
covariancia = np.cov(taxas_ambev, taxas_itau)[0, 1]
correlacao = np.corrcoef(taxas_ambev, taxas_itau)[0, 1]

print(f"Covariância: {covariancia}, Correlação: {correlacao}")
```

## Aulas de Machine Learning

### 9. Previsão de Preços de Ações com Regressão Linear
Neste script, implementamos um modelo de regressão linear para prever os preços futuros de uma ação com base em dados históricos.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Definindo variáveis independentes e dependentes
X = dataset[['Open', 'High', 'Low', 'Volume']]
y = dataset['Close']

# Dividindo o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Criando o modelo de regressão linear
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Realizando previsões
y_pred = modelo.predict(X_test)

# Avaliando o desempenho do modelo
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
```

### 10. Classificação de Tendência com Machine Learning
Implementamos um modelo de classificação para prever se o preço de uma ação terá uma tendência de alta ou baixa com base em indicadores técnicos.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Criando variável alvo (tendência)
dataset['Trend'] = np.where(dataset['Close'].shift(-1) > dataset['Close'], 1, 0)

# Removendo valores NaN
dataset.dropna(inplace=True)

# Definindo variáveis independentes e dependentes
X = dataset[['Open', 'High', 'Low', 'Close', 'Volume']]
y = dataset['Trend']

# Dividindo o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Criando o modelo de classificação (Random Forest)
modelo_classificacao = RandomForestClassifier()
modelo_classificacao.fit(X_train, y_train)

# Realizando previsões
y_pred = modelo_classificacao.predict(X_test)

# Avaliando a acurácia do modelo
acuracia = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {acuracia * 100:.2f}%')

# Exibindo o relatório de classificação
print(classification_report(y_test, y_pred))
```

Estes scripts servem como material complementar ao curso, permitindo a prática dos conceitos apresentados. Sinta-se à vontade para explorar e adaptar conforme necessário. 
