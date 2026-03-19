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
#tiene uno para cada mes de 1991 a 2008

print(df)

fig,ax = plt.subplots(figsize=(10,8))

ax.plot(df.y)

ax.set_xlabel('Date')
ax.set_ylabel('Number of antidiabetic drug prescriptions')

plt.xticks(np.arange(6,203,12), np.arange(1992,2009))

fig.autofmt_xdate()
plt.tight_layout()

plt.savefig('original.png', dpi=300)

decomposition = STL(df.y, period=12).fit()

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10,8))

ax1.plot(decomposition.observed)
ax1.set_ylabel('Observed')

ax2.plot(decomposition.trend)
ax2.set_ylabel('Trend')

ax3.plot(decomposition.seasonal)
ax3.set_ylabel('Seasonal')

ax4.plot(decomposition.resid)
ax4.set_ylabel('Residuals')

plt.xticks(np.arange(6, 203, 12), np.arange(1992, 2009))

fig.autofmt_xdate()
plt.tight_layout()

plt.savefig('decomposition.png', dpi=300)

print(adfuller(df.y))

df_diff = np.diff(df.y, n=1)

print(adfuller(df_diff))

df_seasonal = np.diff(df_diff, n=12)

print(adfuller(df_seasonal))


def optimize_SARIMAX(endog: Union[pd.Series, list], exog: Union[pd.Series, list], order_list: list, d: int, D:int, s:int) -> pd.DataFrame:

    results = []

    a = tqdm(order_list)

    for p, q, P, Q in a:

        try:
            model = SARIMAX(
                endog=endog, 
                exog=exog,
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

p = range(0,5,1)
q = range(0,5,1)
P = range(0,5,1)
Q = range(0,5,1)


parameter_list = list(product(p,q,P,Q))

d = 1
D = 1
s = 12

train = df.y[:168]
test = df.y[168:]

result = optimize_SARIMAX(train, None, parameter_list, d, D, s)

result.to_csv('result.csv', index=True)
