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

fig, axes = plt.subplots(nrows=3, ncols=2, dpi=300, figsize=(11,6))

for i, ax in enumerate(axes.flatten()[:6]):
    data = macro_econ_data[macro_econ_data.columns[i+2]]
    
    ax.plot(data, color='black', linewidth=1)
    ax.set_title(macro_econ_data.columns[i+2])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines['top'].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.setp(axes, xticks=np.arange(0, 208, 8), xticklabels=np.arange(1959, 2010, 2))
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig('data.png', dpi=300)


target = macro_econ_data['realgdp']

exog = macro_econ_data[['realcons','realinv','realgovt','realdpi', 'cpi']]

print(adfuller(target))

target_diff = target.diff()

print(adfuller(target_diff[1:]))


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


p = range(0,4,1)
q = range(0,4,1)
P = range(0,4,1) 
Q = range(0,4,1) 

d = 1
D = 0

s = 4

parameter_list = list(product(p,q,P,Q))

target_train = target[:200]
exog_train = exog[:200]

result_df = optimize_SARIMAX(target_train, exog_train, parameter_list, d, D, s)

result_df.to_csv('sarimax_results.csv', index=True)






