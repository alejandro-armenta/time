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

print(adfuller(df['Passengers']))

print(df)
print(len(df))

df_diff = np.diff(df['Passengers'], n=1)

print(len(df_diff))
print(adfuller(df_diff))

df_diff_seasonal_diff = np.diff(df_diff, n=12)

print(len(df_diff_seasonal_diff))
print(adfuller(df_diff_seasonal_diff))

p = range(0,4,1)
q = range(0,4,1)
P = range(0,4,1)
Q = range(0,4,1)

d = 1
D = 1

s = 12

order_list = list(product(p,q,P,Q))

train = df['Passengers'][:-12]

def optimize_SARIMA(endog: Union[pd.Series, list], order_list: list, d: int, D:int, s:int) -> pd.DataFrame:

    results = []

    a = tqdm(order_list)

    for p, q, P, Q in a:

        try:
            model = SARIMAX(
                endog, 
                order=(p, d, q),
                seasonal_order=(P, D, Q, s),
                simple_differencing=False
                ).fit(disp=False)
            
        except:
            continue

        aic = model.aic

        results.append([(p, d, q, P, D, Q, s), aic])
    
    result_df = pd.DataFrame(results)

    result_df.columns = ['(p,d,q,P,D,Q,s)', 'AIC']

    result_df.sort_values(by='AIC', ascending=True, inplace=True)

    result_df.reset_index(drop=True, inplace=True)

    return result_df

Sarima_result_df = optimize_SARIMA(train, order_list, d, D, s)

Sarima_result_df.to_csv('sarima_results.csv', index=True)
