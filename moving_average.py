import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('data/widget_sales.csv')

#print(df.describe())

fig, ax = plt.subplots()

ax.plot(df['widget_sales'])
ax.set_xlabel('Time')
ax.set_ylabel('Widget sales (k$)')

plt.xticks(
    [0, 30, 57, 87, 116, 145, 175, 204, 234, 264, 293, 323, 352, 382, 409, 439, 468, 498], 
    ['Jan 2019', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan 2020', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])

fig.autofmt_xdate()
plt.tight_layout()
plt.savefig('widget_sales.png', dpi=300)

#print(adfuller(df['widget_sales']))

widget_sales_diff = np.diff(df['widget_sales'], n=1)

#print(widget_sales_diff)

#ya es estacionario
#print(adfuller(widget_sales_diff))

plot_acf(widget_sales_diff, lags=30)

#ma(2)
plt.savefig('acf_widget_sales_diff.png',dpi=300)

df_diff = pd.DataFrame(
    {
        'widget_sales_diff':widget_sales_diff
    }
)

#print(df_diff)

train_size = int(0.9 * len(df_diff))

train = df_diff[:train_size]
test = df_diff[train_size:]

fig, (ax1, ax2) = plt.subplots(nrows=2,ncols=1,sharex=True)

ax1.plot(df['widget_sales'])
ax1.set_ylabel('Widget sales (k$)')
ax1.axvspan(450, 499, color='#808080', alpha=0.2)

ax2.plot(df_diff['widget_sales_diff'])
ax2.set_xlabel('Time')
ax2.set_ylabel('Derivative Widget sales')
ax2.axvspan(449, 498, color='#808080', alpha=0.2)

plt.xticks(
    ticks=[0, 30, 57, 87, 116, 145, 175, 204, 234, 264, 293, 323, 352, 382, 409, 439, 468, 498],
    labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', '2020', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
)

fig.autofmt_xdate()
plt.tight_layout()

def rolling_forecast(
        df:pd.DataFrame, 
        train_len:int, 
        horizon:int, 
        window:int, 
        method:str) -> list:
    
    total_length = train_len + horizon    

    if method == 'mean':
        pred_mean = []

        #si esta bien hecho
        for i in range(train_len, total_length, window):
            #este mean es para este y el que sigue
            mean = np.mean(df[:i].values)
            pred_mean.extend(mean for _ in range(window))
        
        #print(len(pred_mean))

        return pred_mean
    
    elif method == 'last':
        pred_last_value = []

        #si esta bien hecho
        for i in range(train_len, total_length, window):
            last_value = df[:i].iloc[-1].values[0]

            pred_last_value.extend(last_value for _ in range(window))
        
        return pred_last_value
    
    elif method == 'MA':
        pred_MA = []

        #si esta bien hecho
        for i in range(train_len, total_length, window):

            #en el primero toma solo el training y ya y despues va agregando mas del original

            #se va de 2 en 2 prediciendo pero agarra el full size 

                                       #p,d,q
            model = SARIMAX(df[:i], order=(0,0,2))

            res = model.fit()

            #i ya es el predecido 1 + 1 es el predecido 2
            #window esta afuera de las predicciones
            predictions = res.get_prediction(0, i+window-1)

            oos_pred = predictions.predicted_mean.iloc[-window:]

            pred_MA.extend(oos_pred)

        #print(len(pred_MA))
        

        return pred_MA

pred_df = test.copy()


TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 2

pred_mean = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, method='mean')
pred_last_value = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, method='last')
pred_MA = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, method='MA')

pred_df['pred_mean'] = pred_mean
pred_df['pred_last_value'] = pred_last_value
pred_df['pred_MA'] = pred_MA

#print(pred_df)

#print(mean_squared_error(pred_df['widget_sales_diff'],pred_df['pred_mean']))
#print(mean_squared_error(pred_df['widget_sales_diff'],pred_df['pred_last_value']))

#el ma es el mejor!
#print(mean_squared_error(pred_df['widget_sales_diff'],pred_df['pred_MA']))


fig, ax = plt.subplots()

ax.plot(df_diff['widget_sales_diff'])
ax.plot(pred_df['widget_sales_diff'], 'b-', label='actual')
ax.plot(pred_df['pred_mean'], 'g:', label='mean')
ax.plot(pred_df['pred_last_value'], 'r-.', label='last_value')
ax.plot(pred_df['pred_MA'], 'k--', label='MA(2)')

ax.axvspan(449,498, color='#808080',alpha=0.2)
ax.set_xlim(430,498)

ax.legend(loc=2)

ax.set_xlabel('Time')
ax.set_ylabel('Derivate of Widget sales')

plt.xticks(
    [439, 468, 498], 
    ['Apr', 'May', 'Jun']
    )

plt.tight_layout()
plt.savefig('predictions_differential.png', dpi=300)

#original

df['pred_widget_sales'] = pd.Series()

#este es el de la relidad!

df.loc[450:,'pred_widget_sales'] = df['widget_sales'].iloc[450] + pred_df['pred_MA'].cumsum().values

#print(df.loc[450:,'pred_widget_sales'])

#print(type(pred_df['pred_MA'].cumsum().values))
#print(type(df['widget_sales'].iloc[450]))


#print(df)

plt.close('all')

fig, ax = plt.subplots()

#el diferencial es representativo de ese punto.
#si porque hay una representaicion en ese punto.

#se predice 450 el punto 450 como base y añadir prediccion

#ese punto no se predice

ax.plot(df['widget_sales'], 'b-', label='actual')
ax.plot(df['pred_widget_sales'], 'k--', label='MA(2)')

ax.axvspan(450,499, color='#808080', alpha=0.2)
ax.set_xlim(400,499)

ax.legend(loc=2)

ax.set_xlabel('Time')
ax.set_ylabel('Widget sales (K$)')

plt.xticks(
    [409, 439, 468, 498],
    ['Mar', 'Apr', 'May', 'Jun']
    )

fig.autofmt_xdate()
plt.tight_layout()

plt.savefig('predictions.png', dpi=300)

#450 - 499
#print(pred_df)

print(mean_absolute_error(df['widget_sales'][450:], df['pred_widget_sales'][450:]))

#2320 usd on average error