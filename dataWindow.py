import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tf.random.set_seed(42)

#es por hora!
#son un dia!

#el shift es 24

train_df = pd.read_csv('train.csv', index_col=0)
val_df = pd.read_csv('val.csv', index_col=0)
test_df = pd.read_csv('test.csv', index_col=0)

#print(train_df)
#print(val_df)
#print(test_df)

class DataWindow():

    def __init__(self, input_width, label_width, shift, train_df=train_df, val_df=val_df, test_df=test_df, label_columns=None):

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.label_columns = label_columns

        if label_columns is not None:
            self.label_columns_indices = {name : i for i, name in enumerate(label_columns)}

        self.column_indices = {name: i for i,name in enumerate(train_df.columns)}

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0,input_width)

        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width

        self.labels_slice = slice(self.label_start, None)

        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        
    def split_to_inputs_labels(self, features):
        
        #features   batches * inputs + labels * names

        #son 2 cubos

        inputs = features[:,self.input_slice,:]
        labels = features[:,self.labels_slice,:]

        if self.label_columns is not None:
            
            #batch size * inputs_width * inputs names
            #batch size * labels_width * labels names

            #matriz labels_width * batch_size
            labels = tf.stack(
                [labels[:,:,self.column_indices[name]] for name in self.label_columns],
                axis=-1
            )

        inputs.set_shape([None, self.input_width, None])            
        labels.set_shape([None, self.label_width, None])

        return inputs,labels

    def plot(self, model=None, plot_col='traffic_volume', max_subplots=3):

        plt.figure(figsize=(12,8))

        

        plt.show()
        

        

dw = DataWindow(input_width=1,label_width=1,shift=1,label_columns=['traffic_volume'])

dw.plot()
#print(dw.split_to_inputs_labels())