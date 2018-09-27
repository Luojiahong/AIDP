import tensorflow as tf
import numpy as np

# 1D conv layer
def conv1(inputs,
         nfilters,
         ksize,
         stride=1,
         padding='SAME',
         use_bias=True,
         activation_fn=tf.nn.relu,
         initializer=tf.contrib.layers.variance_scaling_initializer(),
         regularizer=None,
         scope=None,
         reuse=None):
  with tf.variable_scope(scope, reuse=reuse):
    n_in = inputs.get_shape().as_list()[-1]
    weights = tf.get_variable(
              'weights',
              shape=[ksize, n_in, nfilters],
              dtype=inputs.dtype.base_dtype,
              initializer=initializer,
              collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES],
              regularizer=regularizer)
    
    current_layer = tf.nn.conv1d(inputs, weights, stride, padding=padding)
    
    if use_bias:
      biases = tf.get_variable(
              'biases',
              shape=[nfilters,],
              dtype=inputs.dtype.base_dtype,
              initializer=tf.constant_initializer(0.0),
              collections=[tf.GraphKeys.BIASES, tf.GraphKeys.GLOBAL_VARIABLES])
      current_layer = tf.nn.bias_add(current_layer, biases)
    
    if activation_fn is not None:
       current_layer = activation_fn(current_layer)
    
    return current_layer


# 2D conv
def conv2(inputs,
         nfilters,
         ksize, # 2d list
         strides=[1,1,1,1],
         padding='SAME',
         use_bias=True,
         activation_fn=tf.nn.relu,
         initializer=tf.contrib.layers.variance_scaling_initializer(),
         regularizer=None,
         scope=None,
         reuse=None):
  with tf.variable_scope(scope, reuse=reuse):
    in_chns = inputs.get_shape().as_list()[-1]
    weights = tf.get_variable(
              'weights',
              shape=[ksize[0], ksize[1], in_chns, nfilters],
              dtype=inputs.dtype.base_dtype,
              initializer=initializer,
              collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.GLOBAL_VARIABLES],
              regularizer=regularizer)
    
    current_layer = tf.nn.conv2d(inputs, weights, strides, padding=padding)
    
    if use_bias:
       biases = tf.get_variable(
                'biases',
                shape=[nfilters,],
                dtype=inputs.dtype.base_dtype,
                initializer=tf.constant_initializer(0.0),
                collections=[tf.GraphKeys.BIASES, tf.GraphKeys.GLOBAL_VARIABLES])
       current_layer = tf.nn.bias_add(current_layer, biases)
    
    if activation_fn is not None:
       current_layer = activation_fn(current_layer)
    
    return current_layer


def fc(inputs, 
       nfilters, 
       use_bias=True, 
       activation_fn=tf.nn.relu,
       initializer=tf.contrib.layers.variance_scaling_initializer(),
       regularizer=None, 
       scope=None, 
       reuse=None):
  with tf.variable_scope(scope, reuse=reuse):
    n_in = inputs.get_shape().as_list()[-1] # number of chns in
    weights = tf.get_variable(
      'weights',
      shape=[n_in, nfilters],
      dtype=inputs.dtype.base_dtype,
      initializer=initializer,
      regularizer=regularizer)
    
    current_layer = tf.matmul(inputs, weights)
    
    if use_bias:
      biases = tf.get_variable(
        'biases',
        shape=[nfilters,],
        dtype=inputs.dtype.base_dtype,
        initializer=tf.constant_initializer(0.0))
      current_layer = tf.nn.bias_add(current_layer, biases)
    
    if activation_fn is not None:
      current_layer = activation_fn(current_layer)
  
  return current_layer


def gru(inputs, 
        num_layers=2, 
        num_units = 50,
        batch_size = 32,
        scope=None,
        reuse=None):
  """multi-layer GRU"""
  with tf.variable_scope(scope, reuse=reuse):
    rnn_cell = tf.contrib.rnn.MultiRNNCell(
        [tf.contrib.rnn.GRUCell(num_units) for _ in range(num_layers)])
#    if is_training:
#      cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=input_dropout, output_keep_prob=output_dropout)
    init_state = rnn_cell.zero_state(batch_size, tf.float32)
    outputs, states = tf.nn.dynamic_rnn(rnn_cell, inputs, initial_state = init_state, dtype=tf.float32)
  return outputs, states

def lstm(inputs,
        num_layers=2,
        num_units = 50,
        batch_size = 32,
        scope=None,
        reuse=None):
  """multi-layer LSTM"""
  with tf.variable_scope(scope, reuse=reuse):
    rnn_cell = tf.contrib.rnn.MultiRNNCell(
        [tf.contrib.rnn.LSTMCell(num_units) for _ in range(num_layers)])
#    if is_training:
#      cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=input_dropout, output_keep_prob=output_dropout)
    init_state = rnn_cell.zero_state(batch_size, tf.float32)
    outputs, states = tf.nn.dynamic_rnn(rnn_cell, inputs, initial_state = init_state, dtype=tf.float32)
  return outputs, states
  
def bi_gru(inputs,
          num_layers = 2,
          num_units = 50,
          batch_size = 32,
#          initializer = tf.contrib.layers.xavier_initializer(), 
          initializer = tf.contrib.layers.variance_scaling_initializer(),
#          initializer = tf.orthogonal_initializer(),
          scope=None,
          reuse=None):
  """bi-dir GRU"""
  with tf.variable_scope(scope, initializer=initializer, reuse=reuse):
    fw_cell = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.GRUCell(num_units) for _ in range(num_layers)])
#    fw_init = fw_cell.zero_state(batch_size, tf.float32)
    bw_cell = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.GRUCell(num_units) for _ in range(num_layers)])
#    bw_init = bw_cell.zero_state(batch_size, tf.float32)
    outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, 
                                                      inputs, dtype=tf.float32)
#                      initial_state_fw=fw_init, initial_state_bw=bw_init, dtype=tf.float32)
  return outputs, states

def bi_lstm(inputs,
          num_layers = 2,
          num_units = 50,
          batch_size = 32,
          scope=None,
          reuse=None):
  """bi-dir LSTM"""
  with tf.variable_scope(scope, reuse=reuse):
    fw_cell = tf.contrib.rnn.MultiRNNCell(
               [tf.contrib.rnn.LSTMCell(num_units) for _ in range(num_layers)])
    fw_init = fw_cell.zero_state(batch_size, tf.float32)
    bw_cell = tf.contrib.rnn.MultiRNNCell(
               [tf.contrib.rnn.LSTMCell(num_units) for _ in range(num_layers)])
    bw_init = bw_cell.zero_state(batch_size, tf.float32)
    outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs,
                      initial_state_fw=fw_init, initial_state_bw=bw_init, dtype=tf.float32)
  return outputs, states

#TODO
def batch_norm(inputs, scope=None, is_training=False):
  with tf.variable_scope(scope):
    return tf.layers.batch_normalization(inputs,
        axis=-1,
        momentum=0.99,
        epsilon=0.001,
        center=True,
        scale=True,
        beta_initializer=tf.zeros_initializer(),
        gamma_initializer=tf.ones_initializer(),
        moving_mean_initializer=tf.zeros_initializer(),
        moving_variance_initializer=tf.ones_initializer(),
        beta_regularizer=None,
        gamma_regularizer=None,
        training=is_training,
        trainable=True,
        name=scope,
        reuse=None,
        renorm=False,
        renorm_clipping=None,
        renorm_momentum=0.99,)

def max_pooling(inputs, ksize=2, stride=2):
  inputs = tf.expand_dims(inputs,2)
  outputs = tf.nn.max_pool(inputs, 
                          ksize=[1, ksize, 1, 1], 
                          strides=[1, stride, 1, 1], 
                          padding='SAME')
  outputs = tf.squeeze(outputs,squeeze_dims=[2])
  return outputs
