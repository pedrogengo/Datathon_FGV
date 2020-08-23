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