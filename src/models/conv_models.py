import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential

class ModelConstruction:
    def __init__(self,model_name,data_def):
        self.model_name = model_name
        self.audio_length = data_def.audio_length
        self.n_classes = data_def.n_classes

    def model(self, kernel_size=80):
        if self.model_name == 'm3':
            return self.__m3(kernel_size)
        elif self.model_name == 'm5':
            return self.__m5(kernel_size)
        else:
            exit('model '+self.model_name+' did not found. Choose from [m3, m5]')

    def __m3(self, kernel_size):
        print('model M3')
        m = Sequential()
        m.add(Conv1D(256,
                    input_shape=[self.audio_length, 1],
                    kernel_size=kernel_size,
                    strides=4,
                    padding='same',
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=regularizers.l2(l=0.0001)))
        m.add(BatchNormalization())
        m.add(Activation('relu'))
        m.add(MaxPooling1D(pool_size=4, strides=None))
        m = self.__add_intermediate_convolution_block(m, 256)
        m.add(Lambda(lambda x: K.mean(x, axis=1)))
        m.add(Dense(self.n_classes, activation='softmax'))
        return m

    def __m5(self, kernel_size):
        print('model M5')
        m = Sequential()
        m.add(Conv1D(128,
                    input_shape=[self.audio_length, 1],
                    kernel_size=kernel_size,
                    strides=4,
                    padding='same',
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=regularizers.l2(l=0.0001)))
        m.add(BatchNormalization())
        m.add(Activation('relu'))
        m.add(MaxPooling1D(pool_size=4, strides=None))
        m = self.__add_intermediate_convolution_block(m, 128)
        m = self.__add_intermediate_convolution_block(m, 256)
        m = self.__add_intermediate_convolution_block(m, 512)
        m.add(Lambda(lambda x: K.mean(x, axis=1)))
        m.add(Dense(self.n_classes, activation='softmax'))
        return m

