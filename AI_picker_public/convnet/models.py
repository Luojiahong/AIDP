import os, time

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

import tflib.model
import tflib.layers as layers
import tflib.GRU as GRU

def get(model_name, inputs, config, checkpoint_dir, is_training=False):
  """Returns a Model instance by model name.
  Args:
    name:    class name of the model to instantiate.
    samples: samples as returned by a DataPipeline.
    params:  dict of model parameters.
  """
  return globals()[model_name](inputs, config, checkpoint_dir, is_training)


class DetNet(tflib.model.BaseModel):
  
  def __init__(self, inputs, config, checkpoint_dir, is_training=False,
               reuse=False):
    self.is_training = is_training
    self.config = config
    super(DetNet, self).__init__(inputs, 
                                 checkpoint_dir, 
                                 is_training=is_training,
                                 reuse=reuse)
  
  def _setup_prediction(self):
    # def inputs
    self.batch_size = self.inputs['data'].get_shape().as_list()[0]
    current_layer = self.inputs['data']
    current_layer = tf.squeeze(current_layer, [1])
    
    # def hypo-params
    c     = 32 # number of channels per conv layer
    ksize = 3  # size of the convolution kernel
    depth = 8
    lamb  = 1e-4 # l2 reg
    # def model structure
    for i in range(depth):
        current_layer = layers.conv1(current_layer, c, ksize, stride=1, 
                                    activation_fn=None, scope='conv{}'.format(i+1))
        current_layer = layers.batch_norm(current_layer, 
                                        'conv{}_batch_norm'.format(i+1), 
                                        self.is_training)
        current_layer = tf.nn.relu(current_layer)
        self.layers['conv0_{}'.format(i+1)] = current_layer
        current_layer = layers.max_pooling(current_layer, 2)
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, current_layer)
        self.layers['conv{}'.format(i+1)] = current_layer
    # softmax regression
    bs, width, _ = current_layer.get_shape().as_list()
    current_layer = tf.reshape(current_layer, [bs, width*c], name="reshape")
    
    current_layer = layers.fc(current_layer, 2, scope='det_logits', activation_fn=None)
    self.layers['logits'] = current_layer
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, current_layer)
    # output prediction
    self.layers['class_prob'] = tf.nn.softmax(current_layer, name='det_class_prob')
    self.layers['class_prediction'] = tf.argmax(self.layers['class_prob'], 1, name='det_class_pred')
    
    # add L2 regularization
    tf.contrib.layers.apply_regularization(
        tf.contrib.layers.l2_regularizer(lamb),
        weights_list=tf.get_collection(tf.GraphKeys.WEIGHTS))
  
  # toprint validation messages
  def validation_metrics(self):
    if not hasattr(self, '_validation_metrics'):
      self._setup_loss()
      self._validation_metrics = {
        'loss': self.loss,
        'detection_accuracy': self.detection_accuracy
      }
    return self._validation_metrics
  
  def validation_metrics_message(self, metrics):
    s = 'loss = {:.5f} | det. acc. = {:.1f}%'.format(metrics['loss'], \
         metrics['detection_accuracy']*100)
    return s
  
  # def loss function and accuracy
  def _setup_loss(self):
    with tf.name_scope('loss'):
      targets = self.inputs['label']
      raw_loss = tf.reduce_mean(
                   tf.nn.sparse_softmax_cross_entropy_with_logits(
                     logits = self.layers['logits'], labels = targets))
      self.summaries.append(tf.summary.scalar('loss/train_raw', raw_loss))
    
    self.loss = raw_loss
    
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    if reg_losses:
      with tf.name_scope('regularizers'):
        reg_loss = tf.reduce_sum(reg_losses)
        self.summaries.append(tf.summary.scalar('loss/regularization', reg_loss))
      self.loss += reg_loss
    
    self.summaries.append(tf.summary.scalar('loss/train', self.loss))
    
    with tf.name_scope('accuracy'):
      is_true_event = tf.cast(tf.greater(targets, tf.zeros_like(targets)), tf.int64)
      is_pred_event = tf.cast(tf.greater(self.layers['class_prediction'], 
                                         tf.zeros_like(targets)), tf.int64)
      detection_is_correct = tf.equal(is_true_event, is_pred_event)
      self.detection_accuracy = tf.reduce_mean(tf.to_float(detection_is_correct))
      self.summaries.append(tf.summary.scalar('detection_accuracy/train', self.detection_accuracy))
  
  # def optimizer
  def _setup_optimizer(self, learning_rate):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
      updates = tf.group(*update_ops, name='update_ops')
      with tf.control_dependencies([updates]):
        self.loss = tf.identity(self.loss)
    optim = tf.train.AdamOptimizer(learning_rate).minimize(
        self.loss, name='optimizer', global_step=self.global_step)
    self.optimizer = optim
  
  # def train messages
  def _tofetch(self):
    return {
        'optimizer': self.optimizer,
        'loss': self.loss,
        'detection_accuracy': self.detection_accuracy,
    }
  
  def _summary_step(self, step_data):
    step = step_data['step']
    loss = step_data['loss']
    det_accuracy = step_data['detection_accuracy']
    duration = step_data['duration']
    avg_duration = 1000*duration/step
    
    if self.is_training:
      toprint ='Step {} | {:.0f}s ({:.0f}ms) | loss = {:.4f} | det. acc. = {:.1f}%'.format(
        step, duration, avg_duration, loss, 100*det_accuracy)
    else:
      toprint ='Step {} | {:.0f}s ({:.0f}ms) | accuracy = {:.1f}%'.format(
        step, duration, avg_duration, 100*det_accuracy)
    
    return toprint



class PpkNet(tflib.model.BaseModel):
  
  def __init__(self, inputs, config, checkpoint_dir, 
                is_training=False, reuse=False):
    self.is_training = is_training
    self.config = config
    super(PpkNet, self).__init__(inputs, 
                                 checkpoint_dir, 
                                 is_training = is_training, 
                                 reuse = reuse) 
  
  def _setup_prediction(self):
  
    # RNN params
    self.num_units  = 64  # number of gru cells per layer
    self.num_layers = 2   # number of gru layers
    
    # input data
    self.batch_size = self.inputs['data'].get_shape().as_list()[0]
    current_layer   = self.inputs['data']
    bs, self.num_step, step_len, chn = current_layer.get_shape().as_list()
    current_layer = tf.reshape(current_layer, [bs, self.num_step, -1]) # flatten the chns
    
    # RNN PpkNet
    # gru model
#    output, state = layers.gru(current_layer, num_layers=self.num_layers, num_units=self.num_units, 
#                               batch_size=self.batch_size, scope='multi_gru')
    
    output0, state = layers.bi_gru(current_layer, 
                                  num_layers=self.num_layers, 
                                  num_units =self.num_units, 
                                  batch_size=self.batch_size, 
                                  scope='bi_gru')
    output = tf.concat([output0[0], output0[1]], axis=2)
    """
    bi_gru = GRU.RNN(current_layer, self.num_units)
    output0 = bi_gru.bi_rnn()
    output = tf.concat([output0['state_fw'][1], output0['state_bw'][1]], axis=2)
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, output)
    self.layers['state_bw'] = output0['state_bw']
    self.layers['reset_bw'] = output0['reset_bw']
    self.layers['update_bw']= output0['update_bw']
    self.layers['state_fw'] = output0['state_fw']
    self.layers['reset_fw'] = output0['reset_fw']
    self.layers['update_fw']= output0['update_fw']
    """
    output = tf.reshape(output, [-1, 2*self.num_units]) # flatten bi-rnn
#    output = tf.reshape(output, [-1, self.num_units])   # flatten rnn
    logits = layers.fc(output, 3, scope='ppk_logits', activation_fn=None) # [0 1 2] for [N, P, S] 
    self.layers['logits'] = logits
    tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, logits)
    
    flat_prob = tf.nn.softmax(logits, name='ppk_class_prob')
    self.layers['class_prob'] = tf.reshape(flat_prob, [-1, self.num_step, 3]) # shape=(32, 59, 3)
    self.layers['class_prediction'] = tf.argmax(self.layers['class_prob'], 2, name='ppk_class_pred')
    
  
  def validation_metrics(self):
    if not hasattr(self, '_validation_metrics'):
        self._setup_loss()
        self._validation_metrics = {'loss': self.loss,
                                    'err_rate': self.err_rate}
    return self._validation_metrics
  
  def validation_metrics_message(self, metrics):
      s = 'loss = {:.5f} | det. acc. = {:.1f}'.format(metrics['loss'],
          metrics['err_rate'])
      return s

  def _setup_loss(self):
    with tf.name_scope('loss'):
      # target shape = bs*3*step
      target = self.inputs['target']
      self.targets = tf.argmax(target, 1)
      flat_target = tf.reshape(self.targets, [-1])
      
      # cross entropy loss
      cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(\
                            logits = self.layers['logits'], 
                            labels = flat_target)
      loss = tf.reduce_mean(cross_entropy)
      self.summaries.append(tf.summary.scalar('/train', loss))
      self.loss = loss
    
    with tf.name_scope('err_rate'):
      ppk_is_incorrect = tf.to_float(tf.not_equal(\
                                    self.targets, self.layers['class_prediction']))
      self.err_rate = tf.reduce_sum(ppk_is_incorrect)/self.batch_size
      self.summaries.append(tf.summary.scalar('err_rate/train', self.err_rate))
  
  def _setup_optimizer(self, learning_rate):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
       updates = tf.group(*update_ops, name='update_ops')
       with tf.control_dependencies([updates]):
            self.loss = tf.identity(self.loss)
    optim = tf.train.AdamOptimizer(learning_rate)
    self.optimizer = optim.minimize(self.loss, 
                                    name='optimizer', 
                                    global_step=self.global_step)
  
  def _tofetch(self):
    return { 'optimizer': self.optimizer,
             'loss': self.loss,
             'err_rate': self.err_rate,
             'target': self.targets,
             'pred': self.layers['class_prediction'],
             'org_target': self.inputs['target']
           }
  
  def _summary_step(self, step_data):
    step = step_data['step']
    loss = step_data['loss']
    err_rate = step_data['err_rate']
    duration = step_data['duration']
    avg_duration = 1000*duration /step
    
    if self.is_training:
      toprint ='Step {} | {:.0f}s ({:.0f}ms) | loss = {:.4f} | err_rate = {:.1f}'.\
                format(step, duration, avg_duration, loss, err_rate)
    else:
      toprint = 'Step {} | {:.0f}s ({:.0f}ms) | err_rate = {:.1f}'.\
                format(step, duration, avg_duration, err_rate)
    return toprint
