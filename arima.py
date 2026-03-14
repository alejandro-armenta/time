
#integration order

#integrated series d
#arima(p, d, q)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

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

