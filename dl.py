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



