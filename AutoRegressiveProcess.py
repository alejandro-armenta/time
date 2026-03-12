import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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

