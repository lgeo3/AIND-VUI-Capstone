from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, Conv2D, MaxPooling1D,
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Dropout, Concatenate)
from keras.layers.core import Activation, Reshape

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride, conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, strides=conv_stride, padding=conv_border_mode, activation='relu', name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length( x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29, activation='relu'):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    rnn_input = input_data
    for num in range(recur_layers):
        rnn_name = 'rnn_{}'.format(num)
        simp_rnn = GRU(units, activation=activation, return_sequences=True, implementation=2, name=rnn_name)(rnn_input)
        output = BatchNormalization()(simp_rnn)
        # we set rnn_input to new output, thus allowing to chain rnn
        rnn_input = output


    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(rnn_input)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29, activation='relu'):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    simp_rnn = GRU(units, activation=activation, return_sequences=True, implementation=2)
    bidir_rnn = Bidirectional(simp_rnn)(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model
from keras.engine.topology import Layer

class MyLayerReshape(Layer):

    def __init__(self, **kwargs):
        super(MyLayerReshape, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(MyLayerReshape, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        print("call")
        after_conv = K.reshape(x, (-1, -1,  x.shape[-1].value * x.shape[-2].value))  # reshaping.. so we have [batch_size, time, features]
        return after_conv

    def compute_output_shape(self, input_shape):
        print("Compute")
        return (input_shape[0], input_shape[1], input_shape[-2]* input_shape[-1])
    
class MyLayerExpandDim(Layer):

    def __init__(self, **kwargs):
        super(MyLayerExpandDim, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(MyLayerExpandDim, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        print("call")
        res = K.expand_dims(x)
        return res

    def compute_output_shape(self, input_shape):
        print("Compute")
        return (input_shape[0], input_shape[1], input_shape[2], 1)
    
    
def final_model(input_dim, filters=40, kernel_size=5, units=200, output_dim=29, recur_layers=3,
                activation='relu', dropout_rate=0.1):
    """ Build a deep network for speech
    """
    conv_stride=1
    conv_border_mode='valid'
    # Main acoustic input
    # we add a dimension, to be able to apply convolution1d to frequencies only
    input_data = Input(name='the_input', shape=(None, input_dim))
    
    x = MyLayerExpandDim()(input_data)
    
    
    # applying convolution to frequency domain -> allowing to model spectral variance due to speaker change (better than fullly connected because it preserve orders of frequencies)
    conv_0 = Conv2D(filters, kernel_size, strides=[1, 1], padding=conv_border_mode, activation='relu')(x)
    conv_0 = BatchNormalization()(conv_0)
    conv_1 = Conv2D(filters, kernel_size, strides=[1, 4], padding=conv_border_mode, activation='relu')(conv_0)
    conv_1 = BatchNormalization()(conv_1)
    conv_2 = Conv2D(filters, kernel_size, strides=[1, 4], padding=conv_border_mode, activation='relu')(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    
    #after_conv = K.reshape(conv_2, (-1, -1,  conv_2.shape[-1].value * conv_2.shape[-2].value))  # reshaping.. so we have [batch_size, time, features]
    #after_conv = (tensor=after_conv)
    after_conv = MyLayerReshape()(conv_2)

    
    # fully connected layers that allowed to extract on each timestep complex frequency features
    #fc0 = Dense(filters, activation=activation)(input_data)
    #fc0 = BatchNormalization()(fc0)
    
    #fc1 = Dense(filters//2, activation=activation)(fc0)
    #fc1 = BatchNormalization()(fc1)
    
    #fc2 = Dense(filters//4, activation=activation)(fc1)
    #fc2 = BatchNormalization()(fc2)
    
    # TODO: stack conv1d_0, 1, et 2 pour en faire l'entree du rnn
    rnn_input = after_conv
    #units = filters//4
    #rnn_input = input_data
    for num in range(recur_layers):
        rnn_name = 'rnn_{}'.format(num)
        #  TODO: en ajoutant un dropout en input
        rnn_input = Dropout(dropout_rate)(rnn_input)
        # TODO: tester avec LSTM
        # TODO: tester avec activation ='elu' ou tanh?
        # TODO: tester avec un clipped
        simp_rnn = LSTM(units, activation='tanh', return_sequences=True, implementation=2, name=rnn_name)
        rnn = Bidirectional(simp_rnn)(rnn_input)
        output = BatchNormalization()(rnn)
        # we set rnn_input to new output, thus allowing to chain rnn
        rnn_input = output

    # and finally we add a TimeDistributed
    time_dense = TimeDistributed(Dense(output_dim))(rnn_input)

    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: x
    #model.output_length = lambda x: cnn_output_length( x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model


if __name__ == "__main__":

    # allocate 50% of GPU memory (if you like, feel free to change this)
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf 
    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.9

    # we force the use of xla
    #config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    set_session(tf.Session(config=config))

    # watch for any changes in the sample_models module, and reload it automatically
    # import NN architectures for speech recognition
    from sample_models import *
    # import function for training acoustic model
    from train_utils import train_model
    use_spectrogram = True # we use spectrogram to train all network

    if use_spectrogram:
        input_dim = 161
    else:
        input_dim = 13
        
    # defining a batch size here, 100 seems correct on my machine
    batch_size = 100
    model_end = final_model(input_dim=input_dim)
