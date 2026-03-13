#akaike information criterion (AIC)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import mean_squared_error, mean_absolute_error


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










