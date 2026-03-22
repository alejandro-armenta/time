import tensorflow as tf

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError

from dataWindow import DataWindow, Baseline, MultiStepLastBaseline

import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_csv('train.csv', index_col=0)

#single step baseline 
#son arreglos!
singel_step_window = DataWindow(input_width=1,label_width=1,shift=1,label_columns=['traffic_volume'])

#--!-- 
wide_window = DataWindow(input_width=24,label_width=24,shift=1,label_columns=['traffic_volume'])

        
column_indices = {name:i for i, name in enumerate(train_df.columns)}

baseline_last = Baseline(label_index=column_indices['traffic_volume'])

baseline_last.compile(loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])

val_performance = {}
performance = {}

val_performance['ale'] = baseline_last.evaluate(singel_step_window.val)

performance['ale'] = baseline_last.evaluate(singel_step_window.test)

print(val_performance)
print(performance)


wide_window.plot(baseline_last)
plt.savefig('predictions_baselinge_onestep.png', dpi=300)


multi_window = DataWindow(input_width=24, label_width=24, shift=24, label_columns=['traffic_volume'])


ms_baseline_last = MultiStepLastBaseline(label_index=column_indices['traffic_volume'])

ms_baseline_last.compile(loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])


mae = MeanAbsoluteError()


temp = multi_window.val

y_pred = ms_baseline_last.predict(temp)

y_pred_ = y_pred[:,:,0]

y_pred__ = tf.expand_dims(y_pred_, axis=-1)

print(y_pred__.shape)

#dataset
list_a = list(temp.as_numpy_iterator())

label_list = []
for input,label in list_a:
    label_list.append(label)

a = tf.concat(label_list, 0)
y_true = a

mae.update_state(y_true, y_pred__)

print(mae.result().numpy())

print(ms_baseline_last.evaluate(multi_window.val))

#print(ms_baseline_last.evaluate(multi_window.test))

multi_window.plot(ms_baseline_last)
plt.savefig('predictions_baseline_multistep.png', dpi=300)