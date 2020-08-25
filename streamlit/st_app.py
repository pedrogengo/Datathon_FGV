import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def create_lagged_features(df, lagged_features):
    lagged_features_name = []
    for i in range(lagged_features):
        df['lagged_' + str(i)] = df.close.shift(i)
        if i > 0:
            lagged_features_name.append('lagged_' + str(i))
    return df, lagged_features_name

def train_test_split_standard_norm(df, test_ratio, lagged_features, lagged_features_name):
    final_train_index = int(df.shape[0] * test_ratio)
    X_train = df.iloc[lagged_features:-final_train_index][lagged_features_name]
    y_train = df.iloc[lagged_features:-final_train_index]['close']
    scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    y_train = y_scaler.fit_transform(np.array(y_train).reshape(-1,1))
    X_test  = df.iloc[-final_train_index:][lagged_features_name]
    y_test  = df.iloc[-final_train_index:]['close']
    X_test = scaler.transform(X_test)
    y_test = y_scaler.transform(np.array(y_test).reshape(-1,1))
    return X_train, y_train, X_test, y_test, y_scaler

st.set_option('deprecation.showfileUploaderEncoding', False)


st.title('Cryptocurrency Forecast Analysis')
st.markdown('Your dataset must have the following columns: ["open", "close", "high", "low"]')
st.markdown('The index of your dataset must be a datetime.')
st.markdown('Example of layout:')
example = pd.DataFrame([['2017-08-17', '298', '300', '320', '290'], ['2017-08-18', '300', '302', '320', '290']], columns = ['open_time', 'open', 'close', 'high', 'low'])
example = example.set_index('open_time')
st.dataframe(example)

# datetime_format = st.text_input('Put the datetime format of your dataset here', value='YY-MM-DD')
time_frequency = st.radio("What's the frequency of your data?",
        ('5-Minute', 'Daily', 'Monthly'))
if time_frequency == '5-Minute':
    pass
elif time_frequency == 'Daily':
    pass
elif time_frequency == 'Monthly':
    pass
else:
    pass

crypto_file = None

if time_frequency is not None:
    crypto_file = st.file_uploader('Upload your dataset here', type='csv')

if crypto_file is not None:
    df = pd.read_csv(crypto_file, index_col=0)

    # Visualizacao da serie temporal
    st.subheader('Time series plot')

    fig = px.line(df, x=df.index, y='close', title='Time Series')
    st.plotly_chart(fig, use_container_width=True)

    # Etapa de treino
    st.subheader('Train step')

    final_train_time = st.slider( "Select the final date for the train",
        min_value=datetime.strptime(df.index[0], '%Y-%m-%d'),
        max_value=datetime.strptime(df.index[-1], '%Y-%m-%d'),
        value=datetime.strptime(df.index[int(df.shape[0] * 0.7)], '%Y-%m-%d'),
        format='YYYY-MM-DD')

    # selected_model = st.selectbox('Select a model to use for forecast', ('SVR', 'Linear Regression', 'MLP', 'XGBoost', 'LSTM',
    #                     'HoltWinters', 'Moving Average'))
    selected_model = st.selectbox('Select a model to use for forecast', ('Linear Regression', 'MLP', 'Moving Average'))

    if selected_model == 'Linear Regression':
        lagged_features = st.number_input('How many lagged features do you want to use?', value = 1, step = 1)
        run = st.button('Run')
        if run:
            lagged_features += 1
            df, features = create_lagged_features(df, lagged_features)
            df_train = df[df.index <= final_train_time.strftime("%Y-%m-%d")]
            df_test = df[df.index > final_train_time.strftime("%Y-%m-%d")]
            X_train = df_train.iloc[lagged_features:][features]
            y_train = df_train.iloc[lagged_features:]['close']
            X_test = df_test[features]
            y_test = df_test['close']

            reg = LinearRegression().fit(X_train, y_train)
            y_pred = reg.predict(X_test)
            st.subheader('RMSE = ' + str(round(np.sqrt(mean_squared_error(df_test.close, y_pred)), 6)))
            fig = go.Figure(data = go.Scatter(x=X_test.index, y=y_test, mode='lines', name='real'))
            fig.add_trace(go.Scatter(x=X_test.index, y=y_pred, mode='lines', name='predicted'))
            st.plotly_chart(fig, use_container_width=True)

    if selected_model == 'MLP':
        lagged_features = st.number_input('How many lagged features do you want to use?', value = 1, step = 1)
        layers = st.number_input('How many layers do you want to use?', value = 2, step = 1)
        neurons = st.number_input('How many neurons do you want to use in each layer?', value = 10, step = 1)
        run = st.button('Run')
        if run:
            lagged_features += 1
            df_test = df[df.index > final_train_time.strftime("%Y-%m-%d")]
            test_ratio = df_test.shape[0]/df.shape[0]
            df_lagged, lagged_features_name = create_lagged_features(df, lagged_features)
            X_train, y_train, X_test, y_test, y_scaler = train_test_split_standard_norm(df_lagged, test_ratio, lagged_features, lagged_features_name)
            model_layers = []
            for i in range(layers):
                if i == 0:
                    model_layers.append(tf.keras.layers.Dense(neurons, input_shape=[lagged_features-1], activation="linear"))
                else:
                    model_layers.append(tf.keras.layers.Dense(neurons, activation="linear"))
            model_layers.append(tf.keras.layers.Dense(1, activation='linear'))
            model = tf.keras.models.Sequential(model_layers)
            model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-4, momentum=0.9))
            with st.spinner('Training...'):
                model.fit(X_train, y_train,epochs=200,verbose=0, batch_size=32)
            st.success('Done!')
            forecast = []
            for time in range(y_test.shape[0]-lagged_features+1):
                forecast.append(model.predict(np.array(X_test[time])[np.newaxis]))
            results = np.array(forecast)[:, 0, 0]
            results = y_scaler.inverse_transform(list(results))
            st.subheader('RMSE = ' + str(round(np.sqrt(mean_squared_error(df_test.close, y_pred)), 6)))
            fig = go.Figure(data = go.Scatter(x=df_test.index, y=df_test.iloc[lagged_features:].close, mode='lines', name='real'))
            fig.add_trace(go.Scatter(x=df_test.index, y=results, mode='lines', name='predicted'))
            st.plotly_chart(fig, use_container_width=True)
    
    if selected_model == 'Moving Average':
        window_size = st.number_input('What is the window size?', value = 2, step = 1)
        run = st.button('Run')
        if run:
            window_size += 1
            df_ma, lagged_features_name = create_lagged_features(df, window_size)
            df_ma = df_ma.iloc[window_size-1:]
            df_ma['lagged_sum'] = df[lagged_features_name].sum(axis=1)
            df_ma['y_pred'] = df_ma['lagged_sum']/(window_size-1)
            y_pred = list(df_ma['y_pred'])
            st.subheader('RMSE = ' + str(round(np.sqrt(mean_squared_error(df_ma.close, y_pred)), 6)))
            fig = go.Figure(data = go.Scatter(x=df_ma.index, y=df_ma.close, mode='lines', name='real'))
            fig.add_trace(go.Scatter(x=df_ma.index, y=y_pred, mode='lines', name='predicted'))
            st.plotly_chart(fig, use_container_width=True)
