from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm
from itertools import product
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


df = pd.read_csv('data/air-passengers.csv')

train = df['Passengers'][:-12]
test = pd.read_csv('forecasts.csv', index_col=0)

sarima = SARIMAX(train, order=(2,1,1), seasonal_order=(1,1,2,12), simple_differencing=False)

fit = sarima.fit(disp=False)

fit.plot_diagnostics(figsize=(10,8))
plt.savefig('diagnostics_sarima.png')

print(acorr_ljungbox(fit.resid, np.arange(1,11,1)))

a = fit.get_prediction(132,143).predicted_mean

test['SARIMA_pred'] = a

print(test)

fig,ax = plt.subplots()

ax.plot(df['Month'], df['Passengers'])

ax.plot(test['Passengers'], 'b-', label='actual')
ax.plot(test['naive_seasonal'], 'r:', label='naive seasonal')
ax.plot(test['ARIMA_pred'], 'k--', label='ARIMA(11,2,3)')
ax.plot(test['SARIMA_pred'], 'g-.', label='SARIMA(2,1,1)(1,1,2,12)')

ax.axvspan(132,143, color='grey', alpha=0.2)

ax.legend(loc=2)

plt.xticks(np.arange(0,145,12), np.arange(1949,1962,1))

ax.set_xlim(120,143)
ax.set_xlabel('Time')
ax.set_ylabel('Number of air passengers')

fig.autofmt_xdate()

plt.tight_layout()
plt.savefig('forecasts.png', dpi=300)

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print('MAPE naive seasonal:', mape(test['Passengers'], test['naive_seasonal']))
print('MAPE ARIMA:', mape(test['Passengers'], test['ARIMA_pred']))
print('MAPE SARIMA:', mape(test['Passengers'], test['SARIMA_pred']))

