from tensorflow.keras.layers import Input, Conv3D
from tensorflow.keras.models import Model
import numpy as np


class Sobel():
    def __init__(self):
        inputs = Input(shape=((None, None, None, 1)))
        x = Conv3D(filters=2, kernel_size=(1,3,3), padding='valid', use_bias=False)(inputs)
        self.edge_conv = Model(inputs=inputs, outputs=x)

        edge_kx = np.array([[1, 0, -1], 
                            [2, 0, -2], 
                            [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], 
                            [0, 0, 0], 
                            [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = np.transpose(edge_k, (1,2,0)).reshape(1, 3, 3, 1, 2)
        self.edge_conv.layers[1].set_weights([edge_k])
        # self.edge_conv.trainable = False
        # self.edge_conv.compile(optimizer="adam", loss="mse")
        # self.edge_conv.summary()
        # _________________________________________________________________
        # Layer (type)                 Output Shape                 Param #
        # =================================================================
        # input_35 (InputLayer)        (None, None, None, None, 1)  0      
        # _________________________________________________________________
        # conv3d_38 (Conv3D)           (None, None, None, None, 2)  18     
        # =================================================================
        # Total params: 18
        # Trainable params: 0
        # Non-trainable params: 18
        # _________________________________________________________________
    
    def get_gradient(self, x):
        grad = self.edge_conv(x)
        return grad