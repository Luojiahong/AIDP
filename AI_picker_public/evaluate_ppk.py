"""
evaluate DetNet
"""

import argparse, os, time, sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
sys.path.append('/home/zhouyj/Documents/AIDP/')

import numpy as np
import tensorflow as tf

from   convnet.data_pipeline import DataPipeline
import convnet.models as models
import convnet.config as config


def main(args):
    
    # hypo params
    win_size    = 30          # in seconds
    step_len    = 100         # length of each time step (frame size)
    step_stride = step_len/2  # half overlap of time steps
    num_step    = -(step_len/step_stride-1) + win_size*100 /step_stride
    
    cfg = config.Config()
    cfg.batch_size = 256
    cfg.n_epochs = 1
    
    ckpt_num  = int(6e4)
    ckpt_step = 10
    
    
    # output evaluations
    out_file = 'evaluate_Ppk'
#    if os.path.exists(out_file):
#        os.unlink(out_file)
    out = open(out_file, 'a')
    
    
    for step in range(ckpt_step, ckpt_num+ckpt_step, ckpt_step):
        
        # data pipeline
        data_pipeline = DataPipeline(args.dataset, cfg, step_len+1, num_step, 3, is_training=False)
        samples = {
            'data':  data_pipeline.samples,
            'target': data_pipeline.targets
            }
        
        # set up model and validation metrics
        model = models.get(args.model, samples, cfg,
                        os.path.join(args.checkpoint_dir, args.model), 
                        is_training=False)
        model._setup_loss()
        to_fetch = model.err_rate
        
        with tf.Session() as sess: 
            # start evaluating
            coord = tf.train.Coordinator()
            tf.local_variables_initializer().run()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            # load model
            model.load(sess, step)
            
            while True:
                try:
                    ppk_error = []
                    for _ in range(5):
                        ppk_error.append(sess.run(to_fetch))
                    ppk_error = np.array(ppk_error)
                    err_mean = np.mean(ppk_error)
                    err_std  = np.std(ppk_error)
                    
                except KeyboardInterrupt:
                    print('stopping evaluation')
                    break
                
                except tf.errors.OutOfRangeError:
                    break
                
                finally:
                    coord.request_stop()
                
            coord.join(threads)
            print('eval step {}, ppk error = {}, {}'.format(step, err_mean, err_std))
            out.write('{},{},{}\n'.format(step, err_mean, err_std))
            coord.request_stop()
            coord.join(threads)
        tf.reset_default_graph()
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/home/zhouyj/Documents/AIDP/data/test/finetune/ppk_frame100_stride50',
                        help='path to the recrords to evaluate')
    parser.add_argument('--checkpoint_dir',default='/home/zhouyj/Documents/AIDP/output/finetune',
                        type=str, help='path to checkpoints directory')
    parser.add_argument('--model',type=str,default='PpkNet',
                        help='model to load')
    args = parser.parse_args()
    main(args)
