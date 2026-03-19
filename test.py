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

train = df.y[:168]

model = SARIMAX(
    train, order=(3, 1, 1), 
    seasonal_order=(1, 1, 3, 12),
    simple_differencing=False
    )

fit = model.fit(disp=False)

fit.plot_diagnostics(figsize=(10,8))
plt.tight_layout()
plt.savefig('diagnostics.png', dpi=300)

print(acorr_ljungbox(fit.resid, np.arange(1,11,1)))

