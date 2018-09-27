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
    
    win_size = 30
    cfg = config.Config()
    
    num_step  = int(3.5e4)
    ckpt_step = 10
    
    if args.noise:  cfg.batch_size = 128
    if args.events: cfg.batch_size = 256
    cfg.n_epochs = 1
    
    if not args.noise and not args.events:
        raise ValueError("Define if evaluating accuracy on noise or events")
    # Directory in which the evaluation summaries are written
    if args.noise:
        data_dir = os.path.join(args.dataset, 'det_negative')
        out_file = 'evaluate_{}'.format('Noise')
    if args.events:
        data_dir = os.path.join(args.dataset, 'det_positive')
        out_file = 'evaluate_{}'.format('Events')
    
    # output evaluations
    if os.path.exists(out_file):
        os.unlink(out_file)
    out = open(out_file, 'w')
    
    
    for step in range(ckpt_step, num_step+ckpt_step, ckpt_step):
        
        # data pipeline
        data_pipeline = DataPipeline(data_dir, cfg, 100*win_size +1, 1, 3, is_training=False)
        samples = {
            'data':  data_pipeline.samples,
            'label': data_pipeline.labels[:,0]
            }
        
        # set up model and validation metrics
        model = models.get(args.model, samples, cfg,
                        os.path.join(args.checkpoint_dir, args.model), 
                        is_training=False)
        model._setup_loss()
        to_fetch = model.detection_accuracy
        
        with tf.Session() as sess: 
            # start evaluating
            coord = tf.train.Coordinator()
            tf.local_variables_initializer().run()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            # load model
            model.load(sess, step)
            
            while True:
                try:
                    det_accuracy = []
                    for _ in range(5):
                        det_accuracy.append(sess.run(to_fetch))
                    det_accuracy = np.array(det_accuracy)
                    acc_mean = np.mean(det_accuracy)
                    acc_std  = np.std(det_accuracy)
                    
                except KeyboardInterrupt:
                    print('stopping evaluation')
                    break
                
                except tf.errors.OutOfRangeError:
                    break
                
                finally:
                    coord.request_stop()
                
            coord.join(threads)
            print('eval step {}, detect accuracy = {}, {}'.format(step, acc_mean, acc_std))
            out.write('{},{},{}\n'.format(step, acc_mean, acc_std))
            coord.request_stop()
            coord.join(threads)
        tf.reset_default_graph()
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/home/zhouyj/Documents/AIDP/data/test/finetune',
                        help='path to the recrords to evaluate')
    parser.add_argument('--checkpoint_dir',default='/home/zhouyj/Documents/AIDP/output/finetune',
                        type=str, help='path to checkpoints directory')
    parser.add_argument('--model',type=str,default='DetNet',
                        help='model to load')
    parser.add_argument('--noise', action='store_true',
                        help='pass this flag if evaluate acc on noise')
    parser.add_argument('--events', action='store_true',
                        help='pass this flag if evaluate acc on events')
    
    args = parser.parse_args()
    main(args)
