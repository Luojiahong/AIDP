# -------------------------------------------------------------------
# File:    train_ppk.py
# Author:  Yijian Zhou
# Created: 2017-09-15
# -------------------------------------------------------------------
"""Train PpkNet"""

import argparse, os, time, sys
sys.path.append('/home/zhouyj/Documents/AIDP/')
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf

import convnet.models as models
from convnet.data_pipeline import DataPipeline
import convnet.config as config


def main(args):
    tf.set_random_seed(1234)
    
    cfg = config.Config()
    cfg.batch_size = args.batch_size
    
    # hypo params
    win_size    = 30          # in seconds
    step_len    = 100         # length of each time step (frame size)
    step_stride = step_len/2  # half overlap of time steps
    num_step    = -(step_len/step_stride-1) + win_size*100 /step_stride
    
    ppk_pipeline = DataPipeline(args.dataset, cfg, step_len+1, num_step, 3, True)
    samples = {
        'data':   ppk_pipeline.samples,
        'target': ppk_pipeline.targets
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
   parser.add_argument('--model', type=str, default='PpkNet')
   parser.add_argument('--checkpoint_dir', type=str, 
                      default='/home/zhouyj/Documents/AIDP/output/finetune')
   parser.add_argument('--dataset', type=str, 
                      default='/home/zhouyj/Documents/AIDP/data/train/finetune/ppk_frame100_stride50')
   parser.add_argument('--batch_size', type=int, default=64)
   parser.add_argument('--learning_rate', type=float, default=1e-4)
   parser.add_argument('--resume', action='store_true')
   parser.set_defaults(resume=False)
   args = parser.parse_args()
   
   args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.model)
   main(args)
