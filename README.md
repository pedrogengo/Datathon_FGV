# Datathon FGV - Grupo: snake_case
## Introdução
Esse repositório foi desenvolvido durante o Datathon da FGV, cujo o tema era criptomoedas.
<br>
Tínhamos como objetivo realizar uma análise, de tema livre, sobre o universo de criptomoedas. Decidimos então realizar o forecast do preço diário e 5-5min de algumas criptomoedas e utilizar a biblioteca [Shap](https://github.com/slundberg/shap) para trazer explicação das variáveis, sendo possível uma análise mais profunda das nossas predições.

## Metodologia
Iniciamos nossa análise pela obtenção da base de dados. Consultamos vários datasets públicos, mas alguns apresentavam inconsistência e outro, muitas vezes, não eram de alta frequência. No entanto, na nossa busca, encontramos uma base muito bem estruturada no [Kaggle](https://www.kaggle.com/jorijnsmit/binance-full-history). Tal coleção traz uma série de bases, minuto a minuto, de pares de criptomoedas. Além disso, o fato de estar na estrutura *parquet* a torna mais leve de se trabalhar.
<br>
Com a base em mãos, consultamos o site [Coin Market Cap](https://coinmarketcap.com/pt-br/) para selecionar com quais moedas iriamos fazer as análises, com base na capitalização de mercado das moedas. Escolhemos então as seguintes moedas:

- Ethereum
- Bitcoin
- Ripple
- Iota
- Litecoin

Decidimos também que gostaríamos de olhar as moedas com seus valores baseado em dólar e em bitcoin, para realizar uma análise de como as escolha do par influencia na predição.
<br>
Em relação a modelagem, decidimos por utilizar desde modelos mais simples até alguns mais complexos, para entender como essas criptomoedas se comportam em cada uma deles e tentar explicar os motivos de termos tido bons ou maus resultados. Os modelos que escolhemos para avaliar foram:

- Support Vector Machines for Regression (SVR)
- Moving Average
- Linear Regression
- Linear Regression with penalties (Ridge and Lasso)
- Multi Layer Preceptron (MLP)
- Recurrent Neural Networks with LSTM blocks
- XGBoost

A ideia de trazer essa análise veio da leitura do artigo [Statistical and Machine Learning forecasting methods: Concerns and ways forward] e resolvemos agregar a questão da explicabilidade, pois sabemos que mais importante do que bons resultados, é entender a forma como nosso modelo realiza suas predições, a fim de encontrar bons insights, garantir a qualidade do processo e até mesmo refiná-lo.

### Validação dos modelos
Vamos dividir esse tópico em duas partes:
- [Conjunto de teste](#conjunto_teste)
- [Predição](#predicao)

#### <a name="conjunto_teste"></a>Conjunto de teste
Um importante ponto na hora de criar um conjunto de teste para séries temporais é garantir que você realize a predição usando valores isolados no tempo, ou seja, treino no passado e realize predições no futuro.
É muito importante que, dada a sazonalidade de uma séria temporal, seja respeitado o limite temporal de tal sazonalidade para realizar a divisão de treino e teste, para que o modelo utilizado, caso consiga, aprenda essa a sazonalidade de série. Quando quebramos tal padrão na hora de dividir nossa base podemos fazer com que o modelo não aprenda isso, ocasionando possíveis erros.

#### <a name="predicao"></a>Predição

### Métricas de avaliação
Para avaliar nossos modelos, utilizamos como métrica de avaliação a raiz da soma dos erros quadrados, também conhecido como RMSE (*Root Mean Squared Error*). Sua fórmula é dada por:
<br>
<br>
<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?RMSE%20%3D%20%5Csqrt%7B%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%28%5Chat%7By%7D_t%20-%20y_t%29%5E2%7D%7BN%7D%7D" />
</p>
<br>
<br>

### Resultados

### Webapp
Desenvolvemos também um webapp para que você possa realizar alguns testes com sua própria base.

<p align="center">
  <img src="imgs/webapp_streamlit.png" />
</p>

Para rodá-lo localmente você deve seguir os seguintes passos (seu desenvolvimento foi utilizando python 3.7):

1. Baixar as dependências do projeto:
<pre>
pip install -r streamlit/requirements.txt
</pre>

2. Executar o app (assumindo que você no mesmo diretório do README.md):
<pre>
streamlit run streamlit/st_app.py
</pre>

3. Acessar o endereço local que irá aparecer no seu terminal.