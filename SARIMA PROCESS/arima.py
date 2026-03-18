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

ARIMA_model = SARIMAX(train, order=(11,2,3), simple_differencing=False)

fit = ARIMA_model.fit(disp=False)

fit.plot_diagnostics(figsize=(10,8))
plt.tight_layout()
plt.savefig('diagnostics_arima.png')

print(acorr_ljungbox(fit.resid, np.arange(1,11,1)))

#this means that the arima model is not capturing all the information from the data

test = df[-12:]

test['naive_seasonal'] = df['Passengers'][120:132].values


a = fit.get_prediction(132,143).predicted_mean

test['ARIMA_pred'] = a

print(test)

test.to_csv('forecasts.csv', index=True)

