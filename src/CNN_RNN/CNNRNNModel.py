""" Construct the RNN model for deep learning
"""
import keras
from keras.models import Sequential
from keras.layers import Bidirectional
from keras.layers import Activation
from keras.layers import LSTM,Masking,Embedding
from keras.optimizers import Adam
from keras import regularizers
from keras.layers import Dense,Activation,Dropout,Flatten
from keras.layers import Conv1D,MaxPooling1D

def CNN_RNN_model():
    SEG_LENTH = 180
    model = Sequential()
    #first layer of convolution and max-pooling
    model.add(Conv1D(16,4,activation = 'tanh',padding = 'same',\
              input_shape = (SEG_LENTH,12)))
    model.add(MaxPooling1D(pool_size = 2))
    #second layer of convolution and max-pooling
    model.add(Conv1D(32,5,activation = 'tanh',padding = 'same'))
    model.add(MaxPooling1D(pool_size = 2))
    #third layer of convolution and max-pooling
    model.add(Conv1D(64,6,activation = 'tanh',padding = 'same'))
    model.add(MaxPooling1D(pool_size = 2))

    #model.add(Masking(mask_value= [0,0,0,0,0,0,0,0,0,0,0,0],\
    #         input_shape=(SEG_LENTH, 12)))
    model.add(LSTM(128,dropout=0.2, recurrent_dropout=0.2,\
              return_sequences = True))
    model.add(LSTM(64,dropout=0.2, recurrent_dropout=0.2,\
                           return_sequences = True))
    model.add(LSTM(2))
    model.add(Activation('softmax'))
    adam = Adam()
    model.compile(loss = 'categorical_crossentropy',optimizer = adam,\
              metrics = ['accuracy'])
    print(model.summary())
    return model 

if __name__ == "__main__":
     model = CNN_RNN_model()
