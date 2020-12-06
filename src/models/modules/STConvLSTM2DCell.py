from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import array_ops
from tensorflow.keras.layers import BatchNormalization, Conv2D


class RefineNet(Layer):
    def __init__(self, hidden_states_chs=8):
        super(RefineNet, self).__init__()
        self.hidden_states_chs = hidden_states_chs
        self.kernel_initializer = 'glorot_uniform'
        self.kernel_regularizer = None
        self.kernel_constraint = None
        self.bias_initializer = 'zeros'
        self.bias_regularizer = None
        self.bias_constraint = None

    def build(self, input_shape):
        input_chs = input_shape[-1]    # CLSTM's input_chs + hidden_states_chs
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.refine_1_kernel = self.add_weight(name='refine_1_kernel',
                        shape=(5, 5, 
                                input_chs, 
                                input_chs),
                        initializer=self.kernel_initializer,
                        regularizer=self.kernel_regularizer,
                        constraint=self.kernel_constraint)
        self.refine_2_kernel = self.add_weight(name='refine_2_kernel',
                        shape=(5, 5, 
                                input_chs, 
                                input_chs),
                        initializer=self.kernel_initializer,
                        regularizer=self.kernel_regularizer,
                        constraint=self.kernel_constraint)
        self.refine_h_kernel = self.add_weight(name='refine_h_kernel',
                        shape=(3, 3, 
                                input_chs, 
                                self.hidden_states_chs),
                        initializer=self.kernel_initializer,
                        regularizer=self.kernel_regularizer,
                        constraint=self.kernel_constraint)
        self.refine_d_kernel = self.add_weight(name='refine_d_kernel',
                        shape=(5, 5, 
                                input_chs, 
                                1),
                        initializer=self.kernel_initializer,
                        regularizer=self.kernel_regularizer,
                        constraint=self.kernel_constraint)
        self.refine_1_bias = None
        self.refine_2_bias = None
        self.refine_h_bias = self.add_weight(name='refine_h_bias',
                        shape=(self.hidden_states_chs,),
                        initializer=self.bias_initializer,
                        regularizer=self.bias_regularizer,
                        constraint=self.bias_constraint)
        self.refine_d_bias = self.add_weight(name='refine_d_bias',
                        shape=(1,),
                        initializer=self.bias_initializer,
                        regularizer=self.bias_regularizer,
                        constraint=self.bias_constraint)
        super(RefineNet, self).build(input_shape)

    def call(self, inputs):
        r_1 = self._conv(inputs, self.refine_1_kernel, self.refine_1_bias, padding='same')
        r_1 = self._BN_relu(r_1, self.bn1)
        r_2 = self._conv(r_1,    self.refine_2_kernel, self.refine_2_bias, padding='same')
        r_2 = self._BN_relu(r_2, self.bn2)
        r_h = self._conv(r_2,    self.refine_h_kernel, self.refine_h_bias, padding='same')
        r_d = self._conv(r_2,    self.refine_d_kernel, self.refine_d_bias, padding='same')
        return r_h, r_d

    def _conv(self, x, w, b=None, padding='same'):
        conv_out = K.conv2d(x, w, strides=(1, 1),
                                padding=padding,
                                data_format='channels_last')
        if b is not None:
            conv_out = K.bias_add(conv_out, b,
                                data_format='channels_last')
        return conv_out

    def _BN_relu(self, x, BN):
        x = BN(x)
        x = K.relu(x)
        return x

    def get_config(self):
        base_config = super(RefineNet, self).get_config()
        out_config = {
            **base_config,
            "hidden_states_chs": self.hidden_states_chs, 
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "kernel_constraint": self.kernel_constraint,
            "bias_initializer": self.bias_initializer,
            "bias_regularizer": self.bias_regularizer,
            "bias_constraint": self.bias_constraint,
        }
        return out_config


class STConvLSTM2DCell(DropoutRNNCellMixin, Layer):
  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1),
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               **kwargs):
    super(STConvLSTM2DCell, self).__init__(**kwargs)
    self.filters = filters
    self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
    self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2,
                                                    'dilation_rate')
    self.activation = activations.get(activation)
    self.recurrent_activation = activations.get(recurrent_activation)
    self.use_bias = use_bias

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.recurrent_initializer = initializers.get(recurrent_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.unit_forget_bias = unit_forget_bias

    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)

    self.kernel_constraint = constraints.get(kernel_constraint)
    self.recurrent_constraint = constraints.get(recurrent_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.dropout = min(1., max(0., dropout))
    self.recurrent_dropout = min(1., max(0., recurrent_dropout))

  @property
  def state_size(self):
    return [self.filters, self.filters]

  def build(self, input_shape):

    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    if input_shape[channel_axis] is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    self.input_dim = input_shape[channel_axis]
    kernel_shape = self.kernel_size + (self.input_dim, self.filters * 3 + self.input_dim)
    self.kernel_shape = kernel_shape
    recurrent_kernel_shape = self.kernel_size + (self.filters, self.filters * 3 + self.input_dim)

    self.kernel = self.add_weight(shape=kernel_shape,
                                  initializer=self.kernel_initializer,
                                  name='kernel',
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
    self.recurrent_kernel = self.add_weight(
        shape=recurrent_kernel_shape,
        initializer=self.recurrent_initializer,
        name='recurrent_kernel',
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint)

    if self.use_bias:
      if self.unit_forget_bias:

        def bias_initializer(_, *args, **kwargs):
          return K.concatenate([
              self.bias_initializer((self.filters,), *args, **kwargs),
              initializers.get('ones')((self.filters,), *args, **kwargs),
              self.bias_initializer((self.filters,), *args, **kwargs),
              self.bias_initializer((self.input_dim,), *args, **kwargs),
          ])
      else:
        bias_initializer = self.bias_initializer
      self.bias = self.add_weight(
          shape=(self.filters * 3 + self.input_dim,),
          name='bias',
          initializer=bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint)
    else:
      self.bias = None
    
    self.refine_net = RefineNet(hidden_states_chs=self.filters)
    self.built = True

  def call(self, inputs, states, training=None):
    h_tm1 = states[0]  # previous memory state
    c_tm1 = states[1]  # previous carry state

    # dropout matrices for input units
    dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
    # dropout matrices for recurrent units
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
        h_tm1, training, count=4)

    if 0 < self.dropout < 1.:
      inputs_i = inputs * dp_mask[0]
      inputs_f = inputs * dp_mask[1]
      inputs_c = inputs * dp_mask[2]
      inputs_o = inputs * dp_mask[3]
    else:
      inputs_i = inputs
      inputs_f = inputs
      inputs_c = inputs
      inputs_o = inputs

    if 0 < self.recurrent_dropout < 1.:
      h_tm1_i = h_tm1 * rec_dp_mask[0]
      h_tm1_f = h_tm1 * rec_dp_mask[1]
      h_tm1_c = h_tm1 * rec_dp_mask[2]
      h_tm1_o = h_tm1 * rec_dp_mask[3]
    else:
      h_tm1_i = h_tm1
      h_tm1_f = h_tm1
      h_tm1_c = h_tm1
      h_tm1_o = h_tm1

    (kernel_i, kernel_f,
     kernel_c, kernel_o) = array_ops.split(self.kernel, [self.filters,self.filters,self.filters, self.input_dim], axis=3)
    (recurrent_kernel_i,
     recurrent_kernel_f,
     recurrent_kernel_c,
     recurrent_kernel_o) = array_ops.split(self.recurrent_kernel, [self.filters,self.filters,self.filters, self.input_dim], axis=3)

    if self.use_bias:
      bias_i, bias_f, bias_c, bias_o = array_ops.split(self.bias, [self.filters,self.filters,self.filters, self.input_dim])
    else:
      bias_i, bias_f, bias_c, bias_o = None, None, None, None

    x_i = self.input_conv(inputs_i, kernel_i, bias_i, padding=self.padding)
    x_f = self.input_conv(inputs_f, kernel_f, bias_f, padding=self.padding)
    x_c = self.input_conv(inputs_c, kernel_c, bias_c, padding=self.padding)
    x_o = self.input_conv(inputs_o, kernel_o, bias_o, padding=self.padding)
    h_i = self.recurrent_conv(h_tm1_i, recurrent_kernel_i)
    h_f = self.recurrent_conv(h_tm1_f, recurrent_kernel_f)
    h_c = self.recurrent_conv(h_tm1_c, recurrent_kernel_c)
    h_o = self.recurrent_conv(h_tm1_o, recurrent_kernel_o)

    i = self.recurrent_activation(x_i + h_i)
    f = self.recurrent_activation(x_f + h_f)
    c = f * c_tm1 + i * self.activation(x_c + h_c)
    o = self.recurrent_activation(x_o + h_o)
    h = K.concatenate((o, self.activation(c)), -1)
    r_h, r_d = self.refine_net(h)
    return r_d, [r_h, c]

  def input_conv(self, x, w, b=None, padding='valid'):
    conv_out = K.conv2d(x, w, strides=self.strides,
                        padding=padding,
                        data_format=self.data_format,
                        dilation_rate=self.dilation_rate)
    if b is not None:
      conv_out = K.bias_add(conv_out, b,
                            data_format=self.data_format)
    return conv_out

  def recurrent_conv(self, x, w):
    conv_out = K.conv2d(x, w, strides=(1, 1),
                        padding='same',
                        data_format=self.data_format)
    return conv_out

  def get_config(self):
    config = {'filters': self.filters,
              'kernel_size': self.kernel_size,
              'strides': self.strides,
              'padding': self.padding,
              'data_format': self.data_format,
              'dilation_rate': self.dilation_rate,
              'activation': activations.serialize(self.activation),
              'recurrent_activation': activations.serialize(
                  self.recurrent_activation),
              'use_bias': self.use_bias,
              'kernel_initializer': initializers.serialize(
                  self.kernel_initializer),
              'recurrent_initializer': initializers.serialize(
                  self.recurrent_initializer),
              'bias_initializer': initializers.serialize(self.bias_initializer),
              'unit_forget_bias': self.unit_forget_bias,
              'kernel_regularizer': regularizers.serialize(
                  self.kernel_regularizer),
              'recurrent_regularizer': regularizers.serialize(
                  self.recurrent_regularizer),
              'bias_regularizer': regularizers.serialize(self.bias_regularizer),
              'kernel_constraint': constraints.serialize(
                  self.kernel_constraint),
              'recurrent_constraint': constraints.serialize(
                  self.recurrent_constraint),
              'bias_constraint': constraints.serialize(self.bias_constraint),
              'dropout': self.dropout,
              'recurrent_dropout': self.recurrent_dropout}
    base_config = super(STConvLSTM2DCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))