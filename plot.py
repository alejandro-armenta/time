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
import statsmodels.api as sm
import pandas as pd
import numpy as np

df = pd.read_csv('data/AusAntidiabeticDrug.csv')

pred_df = pd.read_csv('predictions.csv', index_col=0)

#print(df)

fig,ax = plt.subplots()

ax.plot(df.y)
ax.plot(pred_df.y, 'b-', label='actual')
ax.plot(pred_df.last_season, 'r:', label='naive seasonal')
ax.plot(pred_df.SARIMA, 'k--', label='SARIMA')

ax.set_xlabel('Date')
ax.set_ylabel('Number of anti-diabetic drug prescriptions')

ax.axvspan(168,203, color='gray', alpha=0.2)

ax.legend(loc=2)

plt.xticks(np.arange(6, 203, 12), np.arange(1992, 2009, 1))

plt.xlim(120, 203)

fig.autofmt_xdate()

plt.tight_layout()
plt.savefig('predictions.png', dpi=300)

print(pred_df)

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(mape(pred_df.y, pred_df.last_season))
print(mape(pred_df.y, pred_df.SARIMA))

