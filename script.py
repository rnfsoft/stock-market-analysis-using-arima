from datetime import datetime as dt
from datetime import timedelta as td

import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import register_matplotlib_converters
from pandas_datareader import data as web
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA

register_matplotlib_converters()

def smape(y_true, y_pred):
    return np.mean((np.abs(y_pred-y_true)*200/(np.abs(y_pred) + np.abs(y_true))))

def arima(symbol):
    df =  web.DataReader(symbol, data_source='yahoo', start=dt.now() - td(days=3*365), end=dt.now())

    size = int(len(df) * 0.8)
    train, test = df[:size], df[size:]

    train_ar = train['Adj Close'].values
    test_ar = test['Adj Close'].values

    history = [x for x in train_ar]
    predictions = []

    for t in range(len(test_ar)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test_ar[t]
        history.append(obs)
        print('predicted=%f, actual=%f' % (yhat, obs))

    error = mean_squared_error(test_ar, predictions)
    print('Test MSE: %.3f' % error) # Mean Sqaured Error
    error2 = smape(test_ar, predictions)
    print('Test SMAPE: %.3f' % error2) # Symmetric Mean Absolute Percentage

    plt.figure(figsize=(12,7))
    plt.title(symbol)
    plt.xlabel('Date')
    plt.ylabel('Adj Close')
    plt.plot(test.index, test['Adj Close'], color='red', marker='x', label='Actual Price')
    plt.plot(test.index, predictions, color='green', marker='o', linestyle='dashed', label='Predicted Price')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    arima('GRPN')
