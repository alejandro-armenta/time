#akaike information criterion (AIC)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot

from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.stats.diagnostic import acorr_ljungbox

from sklearn.metrics import mean_squared_error, mean_absolute_error

from itertools import product
from typing import Union

from tqdm import tqdm

df = pd.read_csv('data/bandwidth.csv')

print(df)

#total de megabits por segundo en una hora!

fig,ax = plt.subplots()

ax.plot(df.hourly_bandwidth)

ax.set_xlabel('Time')
ax.set_ylabel('Hourly bandwidth usage (MBps)')

plt.xticks(    
    np.arange(0, 10000, 730),     
    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', '2020', 'Feb']
    )

fig.autofmt_xdate()
plt.tight_layout()

plt.savefig('bandwidth.png', dpi=300)

print(adfuller(df.hourly_bandwidth))
#no es stacionaria

bandwidth_diff = np.diff(df.hourly_bandwidth)

print(adfuller(bandwidth_diff))
#es estacionaria

plot_acf(bandwidth_diff, lags=20)
plt.tight_layout()
plt.savefig('acf_diff_1.png', dpi=300)

plot_pacf(bandwidth_diff, lags=20)
plt.tight_layout()
plt.savefig('pacf_diff_1.png', dpi=300)

#not ar ma
#stationary arma(p,q) process

df_diff = pd.DataFrame({
    'bandwidth_diff': bandwidth_diff
})

df_diff.to_csv('df_diff.csv', index=True)

print(df_diff)

train = df_diff[:-168]
test = df_diff[-168:]

ps = range(0, 4, 1)
qs = range(0, 4, 1)

order_list = list(product(ps, qs))

#esta diciendo que es una serie o una lista
def optimize_arma(endog: Union[pd.Series, list], order_list: list) -> pd.DataFrame:

    results = []

    a = tqdm(order_list)
    for p, q in a:

        model = SARIMAX(endog, order=(p, 0, q)).fit()

        aic = model.aic

        results.append([(p, q), aic])
    
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q)', 'AIC']

    result_df.sort_values(by='AIC', ascending=True, inplace=True)
    result_df.reset_index(drop=True, inplace=True)

    #print(result_df)

    return result_df
 
a = optimize_arma(train['bandwidth_diff'], order_list)

print(a)

model = SARIMAX(train['bandwidth_diff'], order=(3,0,2), simple_differencing=False)

model_fit = model.fit()

residuals = model_fit.resid

print(residuals)

qqplot(residuals, line='45')
plt.tight_layout()
plt.savefig('qqplot.png', dpi=300)

model_fit.plot_diagnostics(figsize=(10,8))
plt.tight_layout()
plt.savefig('diagnostics.png', dpi=300)

#the residuals are uncorrelated si son mayores a 0.05
print(acorr_ljungbox(residuals, np.arange(1,11,1)))

#print(pvalue)

def rolling_forecast(
        df:pd.DataFrame, 
        train_len:int, 
        horizon:int, 
        window:int, 
        method:str) -> list:
    
    total_length = train_len + horizon    

    if method == 'mean':

        pred_mean = []

        for i in range(train_len, total_length, window):
            mean = np.mean(df[:i].values)
            pred_mean.extend(mean for _ in range(window))

        return pred_mean
    
    elif method == 'last':
        pred_last_value = []

        for i in range(train_len, total_length, window):
            last_value = df[:i].iloc[-1].values[0]
            pred_last_value.extend(last_value for _ in range(window))
        
        return pred_last_value
    
    elif method == 'ARMA':

        pred_ARMA = []

        for i in range(train_len, total_length, window):

                                           #p,d,q
            model = SARIMAX(df[:i], order=(3,0,2))

            res = model.fit()

            predictions = res.get_prediction(0, i + window - 1)

            oos_pred = predictions.predicted_mean.iloc[-window:]

            pred_ARMA.extend(oos_pred)

        return pred_ARMA

pred_df = test.copy()

TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 2

a = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, method='mean')
b = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, method='last')
c = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, method='ARMA')

pred_df['pred_mean'] = a
pred_df['pred_last_value'] = b
pred_df['pred_ARMA'] = c


print(pred_df)

pred_df.to_csv('bandwidth_predictions.csv', index=True)








