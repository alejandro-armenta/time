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

train = df.y[:168]

model = SARIMAX(
    train, order=(3, 1, 1), 
    seasonal_order=(1, 1, 3, 12),
    simple_differencing=False
    )

fit = model.fit(disp=False)

fit.plot_diagnostics(figsize=(10,8))
plt.tight_layout()
plt.savefig('diagnostics.png', dpi=300)

print(acorr_ljungbox(fit.resid, np.arange(1,11,1)))

#todo esta bien

#window 12 months

def rolling_forecast(df: Union[pd.Series, list], train_len: int, horizon:int, window:int, method:str) -> list:

    total_length = train_len + horizon    

    if method == 'last_season':
        pred_last_season = []

        for i in range(train_len, total_length, window):

            last_season = df['y'][i-window:i].values

            pred_last_season.extend(last_season)
        
        return pred_last_season
    
    elif method == 'SARIMA':

        pred_SARIMA = []

        for i in range(train_len, total_length, window):

            model = SARIMAX(
                endog=df['y'][:i], 
                order=(3, 1, 1),
                seasonal_order=(1, 1, 3, 12),
                simple_differencing=False
                )

            res = model.fit()

            predictions = res.get_prediction(0, i + window - 1)

            #estas prediciendo de 12 en 12
            oos_pred = predictions.predicted_mean.iloc[-window:]

            pred_SARIMA.extend(oos_pred)

        return pred_SARIMA


TRAIN_LEN = 168

HORIZON = 36

WINDOW = 12

pred_df = df[168:]

pred_df['last_season'] = rolling_forecast(df, TRAIN_LEN, HORIZON, WINDOW, 'last_season')

pred_df['SARIMA'] = rolling_forecast(df, TRAIN_LEN, HORIZON, WINDOW, 'SARIMA')

print(pred_df)

pred_df.to_csv('predictions.csv', index=True)

