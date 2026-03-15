
#integration order

#integrated series d
#arima(p, d, q)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

from typing import Union

from tqdm import tqdm

from itertools import product

df = pd.read_csv('data/jj.csv')

#print(df)

fig, ax = plt.subplots()

ax.plot(df.date, df.data)

ax.set_xlabel('Time')
ax.set_ylabel('Earnings per share (USD)')

plt.xticks(
    np.arange(0, 81, 8),
    [1960, 1962, 1964, 1966, 1968, 1970, 1972, 1974, 1976, 1978, 1980]
    )

fig.autofmt_xdate()
plt.tight_layout()
plt.savefig('jj.png', dpi=300)


print(adfuller(df.data))

eps_diff = np.diff(df.data, n=2)

print(adfuller(eps_diff))


def optimize_arma(endog: Union[pd.Series, list], order_list: list, d: int) -> pd.DataFrame:

    results = []

    a = tqdm(order_list)

    for p, q in a:

        try:
            model = SARIMAX(
                endog, 
                order=(p, d, q), 
                simple_differencing=False
                ).fit(disp=False)
        except:
            continue

        aic = model.aic

        results.append([(p, d, q), aic])
    
    result_df = pd.DataFrame(results)

    result_df.columns = ['(p,d,q)', 'AIC']

    result_df.sort_values(by='AIC', ascending=True, inplace=True)

    result_df.reset_index(drop=True, inplace=True)

    return result_df
p = range(0,4,1)
q = range(0,4,1)
d = 2

a = list(product(p,q))

print(a)

train = df.data[:-4]

result = optimize_arma(train, a, d)

print(result)

result.to_csv('result.csv', index=True)

"""
"""

#3,2,3

#6, 2, 1


model = SARIMAX(train, order=(3,2,3))
model_fit = model.fit()

model_fit.plot_diagnostics(figsize=(10,8))
plt.tight_layout()
plt.savefig('diagnostics.png', dpi=300)

df_test = acorr_ljungbox(model_fit.resid, np.arange(1,11,1))
print('jung:', (df_test['lb_pvalue'] > 0.05).astype(int).sum() == len(df_test))
print(df_test)

test = df[-4:]

test['naive_seasonal'] = df[76:80]['data'].values

test['ARIMA_pred'] = model_fit.get_prediction(80,83).predicted_mean

print(test)

fig,ax = plt.subplots()

ax.plot(df.date, df.data)
ax.plot(test['data'], 'b-', label='actual')
ax.plot(test['naive_seasonal'], 'r:', label='naive')
ax.plot(test['ARIMA_pred'], 'k--', label='ARIMA(3,2,3)')

ax.set_xlabel('Time')
ax.set_ylabel('Earnings per share (USD)')
ax.legend(loc=2)

ax.axvspan(80, 83, color='#808080', alpha=0.2)

plt.xticks(np.arange(0, 81, 8), [1960, 1962, 1964, 1966, 1968, 1970, 1972, 1974, 1976, 1978, 1980])

ax.set_xlim(60,83)

fig.autofmt_xdate()

plt.tight_layout()

plt.savefig('pred.png', dpi=300)

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(mape(test['data'], test['naive_seasonal']))

print(mape(test['data'], test['ARIMA_pred']))



