import datetime

import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

print(tf.__version__)

print(tf.config.list_physical_devices('GPU'))

#trafico cada hora en la I-94
df = pd.read_csv('data/metro_interstate_traffic_volume_preprocessed.csv')

#seasonality of traffic daily
fig, ax = plt.subplots()

ax.plot(df.traffic_volume)

ax.set_xlabel('Time')
ax.set_ylabel('Traffic volume')

plt.xticks(np.arange(7,400,24), 
           ['Friday', 'Saturday', 'Sunday','Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
)

plt.xlim(0,400)
fig.autofmt_xdate()

plt.tight_layout()
plt.savefig('traffic_daily.png', dpi=300)

#seasonality on temperature yearly
fig, ax = plt.subplots()

ax.plot(df['temp'])

ax.set_xlabel('Time')
ax.set_ylabel('Temperature (K)')

plt.xticks([2239, 10999], [2017, 2018])

fig.autofmt_xdate()
plt.tight_layout()
plt.savefig('temperature_yearly.png', dpi=300)


#seasonality on temperature daily
fig, ax = plt.subplots()

ax.plot(df['temp'])

ax.set_xlabel('Time')
ax.set_ylabel('Temperature (K)')

plt.xticks(np.arange(7,400,24), 
           ['Friday', 'Saturday', 'Sunday','Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
)

fig.autofmt_xdate()
plt.tight_layout()
plt.xlim(0,400)
plt.savefig('temperature_daily.png', dpi=300)


print(df.describe().transpose())

df.drop(['rain_1h', 'snow_1h'], axis=1, inplace=True)

timestamp_s = pd.to_datetime(df['date_time']).map(datetime.datetime.timestamp)

day = 24*60*60

#print(timestamp_s)

df['day_sin'] = (np.sin(timestamp_s / day * 2 * np.pi)).values

df['day_cos'] = (np.cos(timestamp_s / day * 2 * np.pi)).values

df.sample(50).plot.scatter('day_cos','day_sin').set_aspect('equal')
plt.savefig('cos.png',dpi=300)

df.drop(['date_time'], axis=1, inplace=True)

print(df)


