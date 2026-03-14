import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error

test = pd.read_csv('bandwidth_predictions.csv', index_col=0)
df_diff = pd.read_csv('df_diff.csv', index_col=0)

#print(test)
#print(df_diff)

fig,ax = plt.subplots()

ax.plot(df_diff['bandwidth_diff'])
ax.plot(test['bandwidth_diff'], 'b-', label='actual')
ax.plot(test['pred_mean'], 'g:', label='mean')
ax.plot(test['pred_last_value'], 'r-.', label='last')
ax.plot(test['pred_ARMA'], 'k--', label='ARMA(3,2)')

ax.axvspan(9831, 9998, color='#808080', alpha=0.2)

ax.legend(loc=2)

ax.set_xlabel('Time')
ax.set_ylabel('Hourly bandwidth (diff)')

ax.set_xlim(9800, 9998)

plt.xticks(
    [9802, 9850, 9898, 9946, 9994],
    ['2020-02-13', '2020-02-15', '2020-02-17', '2020-02-19', '2020-02-21']
           )
fig.autofmt_xdate()

plt.tight_layout()

plt.savefig('bandwidth_predictions.png', dpi=300)

print(mean_squared_error(test['bandwidth_diff'], test['pred_mean']))
print(mean_squared_error(test['bandwidth_diff'], test['pred_last_value']))
print(mean_squared_error(test['bandwidth_diff'], test['pred_ARMA']))

