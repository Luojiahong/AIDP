# -------------------------------------------------------------------
# File:    train_det.py
# Author:  Michael Gharbi <gharbi@mit.edu>
# Modified by Yijian Zhou 2017-09-12
# Created: 2016-10-25
# -------------------------------------------------------------------
"""Train a model."""

import argparse, os, time, sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
sys.path.append('/home/zhouyj/Documents/AIDP/')
import numpy as np
import tensorflow as tf

import convnet.models as models
import convnet.data_pipeline as dp
import convnet.config as config


def main(args):
  
  tf.set_random_seed(1234)
  
  cfg = config.Config()
  cfg.batch_size = args.batch_size
  
  pos_path = os.path.join(args.dataset,"det_positive")
  neg_path = os.path.join(args.dataset,"det_negative")
  
  # hypo params
  win_size = 30  # in seconds
  
  # data pipeline for positive and negative examples
  pos_pipeline = dp.DataPipeline(pos_path, cfg, 100*win_size +1, 1, 3, True)
  neg_pipeline = dp.DataPipeline(neg_path, cfg, 100*win_size +1, 1, 3, True)
  
  pos_samples = {
    'data':  pos_pipeline.samples,
    'label': pos_pipeline.labels
    }
  neg_samples = {
    'data':  neg_pipeline.samples,
    'label': neg_pipeline.labels
    }
  
  samples = {
    "data":  tf.concat([pos_samples["data"], neg_samples["data"]],  axis=0),
    "label": tf.concat([pos_samples["label"],neg_samples["label"]], axis=0)
    }

  # model
  model = models.get(args.model, samples, cfg, args.checkpoint_dir, is_training=True)

  # train loop
  model.train(
    args.learning_rate,
    resume=args.resume,
    summary_step=10,
    checkpoint_step=10) 

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, default='DetNet')
  parser.add_argument('--checkpoint_dir', type=str, 
                      default='/home/zhouyj/Documents/AIDP/output/test')
  parser.add_argument('--dataset', type=str, 
                      default='/home/zhouyj/Documents/AIDP/data/train/finetune')
  parser.add_argument('--batch_size', type=int, default=128)
  parser.add_argument('--learning_rate', type=float, default=1e-4)
  parser.add_argument('--resume', action='store_true')
  parser.set_defaults(resume=False)
  args = parser.parse_args()

  args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.model)

  main(args)
