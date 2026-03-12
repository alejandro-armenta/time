import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

df = pd.read_csv('data/foot_traffic.csv')

#print(a)

#average weekly foot traffic at the retail store

fig, ax = plt.subplots()

ax.plot(df['foot_traffic'])

ax.set_xlabel('Time')
ax.set_ylabel('Average weekly foot traffic')

plt.xticks(np.arange(0,1000,104), labels=np.arange(2000,2020,2))

#print(np.arange(0,1000,104))
#print(np.arange(2000,2020,2))

fig.autofmt_xdate()

plt.tight_layout()

plt.savefig('average_weekly_foot_traffic.png')

print(adfuller(df['foot_traffic']))
#no es stacionario

foot_traffic_diff = np.diff(df['foot_traffic'])

#print(foot_traffic_diff)

print(adfuller(foot_traffic_diff))
#ya es stacionario

#es una suma
plot_acf(foot_traffic_diff, lags=20)
plt.tight_layout()
plt.savefig('acf_diff_1.png')

#no es ma()

plot_pacf(foot_traffic_diff,lags=20)
plt.tight_layout()
plt.savefig('pacf_diff_1.png')

#ar(3)

#es uno menos!

df_diff = pd.DataFrame({
    'foot_traffic_diff':foot_traffic_diff
})

train = df_diff[:-52]
test = df_diff[-52:]


print(len(train))

#es un año
print(len(test))

plt.close('all')

fig, (ax1, ax2) = plt.subplots(nrows=2,ncols=1, sharex=True, figsize=(10,8))

ax1.plot(df['foot_traffic'])
ax1.set_xlabel('Time')
ax1.set_ylabel('Avg. weekly foot traffic')

ax1.axvspan(948, 999, color='#808080', alpha=0.2)

ax2.plot(df_diff['foot_traffic_diff'])
ax2.set_xlabel('Time')
ax2.set_ylabel('Diff. avg. weekly foot traffic')

ax2.axvspan(947, 998, color='#808080', alpha=0.2)


plt.xticks(np.arange(0,1000,104), labels=np.arange(2000,2020,2))

fig.autofmt_xdate()

plt.tight_layout()

plt.savefig('testing_periods.png')

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
    
    elif method == 'AR':

        pred_AR = []

        for i in range(train_len, total_length, window):

                                           #p,d,q
            model = SARIMAX(df[:i], order=(3,0,0))

            res = model.fit()

            predictions = res.get_prediction(0, i + window - 1)

            oos_pred = predictions.predicted_mean.iloc[-window:]

            pred_AR.extend(oos_pred)

        return pred_AR

TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 1

a = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, method='mean')
b = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, method='last')
c = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, method='AR')

print(len(a))
print(len(b))
print(len(c))

test['pred_mean'] = a
test['pred_last_value'] = b
test['pred_AR'] = c

print(test)

fig, ax = plt.subplots()

ax.plot(df_diff['foot_traffic_diff'])
ax.plot(test['foot_traffic_diff'], 'b-', label='actual')

ax.plot(test['pred_mean'], 'g:', label='mean')
ax.plot(test['pred_last_value'], 'r-.',label='last')
ax.plot(test['pred_AR'], 'k--', label='AR(3)')

ax.set_xlabel('Time')
ax.set_ylabel('Diff. avg. weekly foot traffic')

ax.axvspan(947, 998, color='#808080', alpha=0.2)
ax.set_xlim(920, 998)

ax.legend(loc=2)

plt.xticks([936, 988],[2018, 2019])

fig.autofmt_xdate()

plt.tight_layout()

plt.savefig('predictions.png')


