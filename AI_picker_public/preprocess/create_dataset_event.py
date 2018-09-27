"""
Creates tfrecords dataset of CNN filtered time steps for PpkNet
This is done by running NPSNet on 30s time windows
"""

import os, glob, sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
sys.path.append('/home/zhouyj/Documents/AIDP/')

import numpy as np
import tensorflow as tf
from obspy.core import read

from   convnet.data_pipeline import DataWriter
import convnet.config as cfg
import convnet.models as models


def main():
    # hypo params
    win_size = 30  # in seconds
    
    out_class    = 'train'
    stream_paths = '/data/WC_AItrain/finetune/Events/%s/*Z.SAC' %out_class
    stream_dir   = '/data/WC_AItrain/finetune/Events/%s'        %out_class
    output_dir   = '/home/zhouyj/Documents/AIDP/data/%s/finetune/det_positive' %out_class
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    stream_files = sorted(glob.glob(stream_paths))
    
    
    done_file = []
    for stream_file in stream_files:
        
        # one-day's data in a tfrecord file
        sta, time, aug_idx, chn, _ = stream_file.split('.')
        jday = time[0:7] # events happened on one day
        if [jday, aug_idx] not in done_file: done_file.append([jday, aug_idx])
        else: continue
        
        # Write event waveforms and labels in .tfrecords
        output_name = 'events_' + jday + '_'+aug_idx + ".tfrecords"
        output_path = os.path.join(output_dir, output_name)
        writer = DataWriter(output_path)
        
        # Load stream
        stz_paths = sorted(glob.glob(stream_dir + '/*{}*.{}.BHZ.SAC'.format(jday, aug_idx)))
        
        # for all streams:
        for i, stz_path in enumerate(stz_paths):
            sta, time, aug_idx, chn, _ = stz_path.split('.')
            stx = '.'.join([sta, time, aug_idx, 'BHE', 'SAC'])
            sty = '.'.join([sta, time, aug_idx, 'BHN', 'SAC'])
            stz = '.'.join([sta, time, aug_idx, 'BHZ', 'SAC'])
            if not (os.path.exists(stx) and os.path.exists(sty) and os.path.exists(stz)):
                print 'missing trace!'; continue
            stream = read(stx).extend([read(sty)[0], read(stz)[0]])
            stream.detrend('constant')
            stream.filter('highpass', freq=1.)
            stream.normalize()
            
            # drop bad data
            if stream.max()[0]==0.0 or stream.max()[1]==0.0 or stream.max()[2]==0.0:
                print 'brocken trace!'; continue
            
            # stream info
            n_traces  = len(stream)
            n_samples = len(stream[0].data)
            n_pts = stream[0].stats.sampling_rate * win_size + 1
            lebel, p_time, s_time = 1, stream[0].stats.sac.t0, stream[0].stats.sac.t1 
            # convert to time_steps and write to TFRecord
            if (n_traces == 3) and (n_pts == n_samples):
                
                # three chn data
                xdata = np.float32(stream[0].data)
                ydata = np.float32(stream[1].data)
                zdata = np.float32(stream[2].data)
                st_data = np.array([[xdata, ydata, zdata]])
                 
                # Write tfrecords
                writer.write(st_data, win_size, lebel, p_time, s_time)
                print("+ Creating tfrecords for event {}, idx = {}".format(jday, i))
            else: print("Missing waveform for event: %s" %(jday))
        writer.close()

if __name__ == "__main__": main()
