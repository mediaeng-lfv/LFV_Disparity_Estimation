from tensorflow.keras.layers import Input, Conv2D, Conv3D, ZeroPadding3D, Lambda, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from keras.engine.network import Network
from tensorflow.keras import backend as K

import tensorflow as tf
def allocate_gpu_memory(gpu_number=0):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')

    if physical_devices:
        try:
            print("Found {} GPU(s)".format(len(physical_devices)))
            tf.config.set_visible_devices(physical_devices[gpu_number], 'GPU')
            tf.config.experimental.set_memory_growth(physical_devices[gpu_number], True)
            print("#{} GPU memory is allocated".format(gpu_number))
        except RuntimeError as e:
            print(e)
    else:
        print("Not enough GPU hardware devices available")
allocate_gpu_memory()

def conv3D_branch(x):
    x = Lambda(lambda x: x - K.mean(x, axis=(0,1,2,3)))(x)
    x = ZeroPadding3D(padding=(0, 4, 4))(x)
    for n_filters in [32, 64, 64, 64]:
        x = Conv3D(n_filters, kernel_size=3, padding='valid', activation='relu', kernel_initializer='glorot_uniform')(x)
    x = Lambda(lambda x: K.squeeze(x, axis=1))(x)
    return x

def build_model():
    frames = None
    s_size = None
    # prepare shared layers
    dummy_frame_inputs = Input(shape=((9, s_size, s_size, 3)))
    shared_layer_h = Network(dummy_frame_inputs, conv3D_branch(dummy_frame_inputs))
    shared_layer_v = Network(dummy_frame_inputs, conv3D_branch(dummy_frame_inputs))
    
    # build model
    inputs_h = Input(shape=((frames, 9, s_size, s_size, 3)), name='inputs_h')
    processed_h = TimeDistributed(shared_layer_h, name='shared_3Dconv_branch_h')(inputs_h)
    inputs_v = Input(shape=((frames, 9, s_size, s_size, 3)), name='inputs_v')
    processed_v = TimeDistributed(shared_layer_v, name='shared_3Dconv_branch_v')(inputs_v)
    x = Concatenate()([processed_h, processed_v])
    
    for n_filters in [64, 32, 32, 16]:
        x = TimeDistributed(Conv2D(n_filters, kernel_size=3, padding='same', activation='relu', kernel_initializer='glorot_uniform'))(x)
    x = TimeDistributed(Conv2D(1, kernel_size=3, padding='same', kernel_initializer='glorot_uniform'))(x)
    x = TimeDistributed(Lambda(lambda x: K.squeeze(x, axis=3)))(x)

    return Model(inputs=[inputs_h, inputs_v], outputs=x)