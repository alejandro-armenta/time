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

print(df)

decomposition = STL(df['Passengers'], period=12).fit()

#decomposition.plot()
#plt.show()

fig, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10,8))

ax1.plot(decomposition.observed)
ax1.set_ylabel('Observed')

ax2.plot(decomposition.trend)
ax2.set_ylabel('Trend')

ax3.plot(decomposition.seasonal)
ax3.set_ylabel('Seasonal')

ax4.plot(decomposition.resid)
ax4.set_ylabel('Residuals')

plt.xticks(np.arange(0,145,12), np.arange(1949,1962,1))

fig.autofmt_xdate()

plt.tight_layout()
plt.savefig('decomposition.png')

print(adfuller(df['Passengers']))

df_diff = np.diff(df['Passengers'], n=2)

print(adfuller(df_diff))


p = range(0,13,1)
q = range(0,13,1)

P = [0]
Q = [0]

d = 2
D = 0
s = 12

#13*13
ARIMA_order_list = list(product(p,q,P,Q))

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

train = df['Passengers'][:-12]

#print(df['Passengers'])

result_df = optimize_SARIMA(train, ARIMA_order_list, d, D, s)

result_df.to_csv('ARIMA_result.csv', index=True)

