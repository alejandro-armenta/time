
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

macro_econ_data = sm.datasets.macrodata.load_pandas().data

target = macro_econ_data['realgdp']

exog = macro_econ_data[['realcons','realinv','realgovt','realdpi', 'cpi']]

target_train = target[:200]
exog_train = exog[:200]


model = SARIMAX(
    endog=target_train, 
    exog=exog_train, 
    order=(3, 1, 3),
    seasonal_order=(0, 0, 0, 4),
    simple_differencing=False
    )

fit = model.fit(disp=False)

#print(fit.summary())

#realdpi coefficient is not significant = 0.0103
fit.plot_diagnostics(figsize=(10,8))
plt.savefig('diagnostics.png', dpi=300)

print(acorr_ljungbox(fit.resid, np.arange(1,11,1)))

def recursive_forecast(endog: Union[pd.Series, list], exog: Union[pd.Series, list], train_len: int, horizon:int, window:int, method:str) -> list:

    total_length = train_len + horizon    

    if method == 'last':
        pred_last_value = []

        for i in range(train_len, total_length, window):

            last_value = endog[:i].iloc[-1]

            pred_last_value.extend(last_value for _ in range(window))
        
        return pred_last_value
    
    elif method == 'SARIMAX':

        pred_SARIMAX = []

        for i in range(train_len, total_length, window):

            model = SARIMAX(
                endog=endog[:i], 
                exog=exog[:i], 
                order=(3, 1, 3),
                seasonal_order=(0, 0, 0, 4),
                simple_differencing=False
                )

            res = model.fit()

            predictions = res.get_prediction(i, i + window - 1, exog=exog.iloc[i])

            oos_pred = predictions.predicted_mean.iloc[-window:]

            pred_SARIMAX.extend(oos_pred)

        return pred_SARIMAX
    
target_train = target[:196]
target_test =target[196:]

TRAIN_LEN = len(target_train)
HORIZON = len(target_test)
WINDOW = 1

#2008 full y 2009 3
#print(target_test)

pred_df = pd.DataFrame(
    {
        'actual':target_test
    }
    )

a = recursive_forecast(target, exog, TRAIN_LEN, HORIZON, WINDOW, 'last')
b = recursive_forecast(target, exog, TRAIN_LEN, HORIZON, WINDOW, 'SARIMAX')

pred_df['pred_last_value'] = a
pred_df['pred_SARIMAX'] = b

print(pred_df)

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(mape(pred_df['actual'], pred_df['pred_last_value']))
print(mape(pred_df['actual'], pred_df['pred_SARIMAX']))
