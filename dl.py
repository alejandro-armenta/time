import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

print(tf.__version__)

print(tf.config.list_physical_devices('GPU'))

df = pd.read_csv('data/metro_interstate_traffic_volume_preprocessed.csv')

print(df)

#trafico cada hora en la I-94

#cloud cover during the hour sensores
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
plt.savefig('traffic.png', dpi=300)

#seasonality on temperature
fig, ax = plt.subplots()

ax.plot(df['temp'])

ax.set_xlabel('Time')
ax.set_ylabel('Temperature (K)')

plt.xticks([2239, 10999], [2017, 2018])

fig.autofmt_xdate()
plt.tight_layout()
plt.savefig('temperature.png', dpi=300)
