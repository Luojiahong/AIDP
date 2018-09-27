import abc
import os
import time

import numpy as np
import tensorflow as tf


class BaseModel(object):
  """Base model implementing the training loop and general model interface."""
  __metaclass__ = abc.ABCMeta
  
  def __init__(self, inputs, checkpoint_dir, is_training=False,
               reuse=False):
    """Creates a model mapped to a directory on disk for I/O:
    
    Args:
      inputs: input tensor(s), can be placeholders (e.g. for runtime prediction) or
              a queued data_pipeline.
      checkpoint_dir: directory where the trained parameters will be saved/loaded from.
      is_training: allows to parametrize certain layers differently when training (e.g. batchnorm).
      reuse: whether to reuse weights defined by another model.
    """
    
    self.inputs = inputs
    self.checkpoint_dir = checkpoint_dir
    self.is_training = is_training
    
    self.layers = {}
    self.summaries = []
    self.eval_summaries = []
    
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self._setup_prediction()
    self.saver = tf.train.Saver(tf.global_variables(),max_to_keep=10*10000)
    
  @abc.abstractmethod
  def _setup_prediction(self):
    """Core layers for model prediction."""
    pass

  @abc.abstractmethod
  def _setup_loss(self):
    """Loss function to minimize."""
    pass

  @abc.abstractmethod
  def _setup_optimizer(self, learning_rate):
    """Optimizer."""
    pass
  
  @abc.abstractmethod
  def _tofetch(self):
    """Tensors to run/fetch at each training step.
    Returns:
      tofetch: (dict) of Tensors/Ops.
    """
    pass
  
  def _train_step(self, sess, start_time, run_options=None, run_metadata=None):
    """Step of the training loop.

    Returns:
      data (dict): data from useful for printing in 'summary_step'.
                   Should contain field "step" with the current_step.
    """
    tofetch = self._tofetch()
    tofetch['step'] = self.global_step
    tofetch['summaries'] = self.merged_summaries
    sess.run(self.global_step)
    data = sess.run(tofetch, options=run_options, run_metadata=run_metadata)
    data['duration'] = time.time()-start_time
    return data
  
  def _test_step(self, sess, start_time, run_options=None, run_metadata=None):
    """Step of the training loop.

    Returns:
      data (dict): data from useful for printing in 'summary_step'.
                   Should contain field "step" with the current_step.
    """
    tofetch = self._tofetch()
    tofetch['step'] = self.global_step
    tofetch['is_correct'] = self.is_correct[0]
    data = sess.run(tofetch, options=run_options, run_metadata=run_metadata)
    data['duration'] = time.time()-start_time
    return data

  @abc.abstractmethod
  def _summary_step(self, data):
    """Information form data printed at each 'summary_step'.

    Returns:
      message (str): string printed at each summary step.
    """
    pass

  def load(self, sess, step=None):
    """Loads the latest checkpoint from disk.

    Args:
      sess (tf.Session): current session in which the parameters are imported.
      step: specific step to load.
    """
    if step==None:
      checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_dir)
    else:
      checkpoint_path = os.path.join(self.checkpoint_dir,"model-"+str(step))
    self.saver.restore(sess, checkpoint_path)
    step = tf.train.global_step(sess, self.global_step)
    print 'Loaded model at step {} from snapshot {}.'.format(step, checkpoint_path)

  def save(self, sess):
    """Saves a checkpoint to disk.

    Args:
      sess (tf.Session): current session from which the parameters are saved.
    """
    checkpoint_path = os.path.join(self.checkpoint_dir, 'model')
    if not os.path.exists(self.checkpoint_dir):
      os.makedirs(self.checkpoint_dir)
    self.saver.save(sess, checkpoint_path, global_step=self.global_step)
  
  
  def train(self, learning_rate, resume=False, summary_step=100,
            checkpoint_step=100):
    """Main training loop.
    Args:
      learning_rate (float): global learning rate used for the optimizer.
      resume (bool): whether to resume training from a checkpoint.
      summary_step (int): frequency at which log entries are added.
      checkpoint_step (int): frequency at which checkpoints are saved to disk.
    """
    lr = tf.Variable(learning_rate, name='learning_rate',
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_VARIABLES])
    self.summaries.append(tf.summary.scalar('learning_rate', lr))
    
    # Optimizer
    self._setup_loss()
    self._setup_optimizer(lr)
    
    run_options = None
    run_metadata = None
    
    # Summaries
    self.merged_summaries = tf.summary.merge(self.summaries)
    
    with tf.Session() as sess:
      self.summary_writer = tf.summary.FileWriter(self.checkpoint_dir, sess.graph)
      
      print 'Initializing all variables.'
      tf.local_variables_initializer().run()
      tf.global_variables_initializer().run()
      if resume:
        self.load(sess)
      
      print 'Starting data threads coordinator.'
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      
      print 'Starting optimization.'
      start_time = time.time()
      step=0
      try:
        while not coord.should_stop():  # Training loop
          step_data = self._train_step(sess, start_time, run_options, run_metadata)
          step = step_data['step']
          if step > 0 and step % summary_step == 0:
            """
            # show training process (PpkNet)
            train_target = step_data['target'][0]
            print '0'*np.sum(train_target==0) + '1'*np.sum(train_target==1) + '2'*np.sum(train_target==2)
            train_pred   = step_data['pred'][0]
            print '0'*np.sum(train_pred==0) + '1'*np.sum(train_pred==1) + '2'*np.sum(train_pred==2)
            """
            
            np.set_printoptions(threshold='nan') 
            print self._summary_step(step_data)
            self.summary_writer.add_summary(step_data['summaries'], global_step=step)
          
          # Save checkpoint every 'checkpoint_step'
          if checkpoint_step is not None and (
              step > 0) and step % checkpoint_step == 0:
            print 'Step {} | Saving checkpoint.'.format(step)
            self.save(sess)
      
      except KeyboardInterrupt:
        print 'Interrupted training at step {}.'.format(step)
        self.save(sess)
      
      except tf.errors.OutOfRangeError:
        print 'Training completed at step {}.'.format(step)
        self.save(sess)
      
      finally:
        print 'Shutting down data threads.'
        coord.request_stop()
        self.summary_writer.close()
      
      # Wait for data threads
      print 'Waiting for all threads.'
      coord.join(threads)
      
      print 'Optimization done.'
