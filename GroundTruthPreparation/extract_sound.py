import h5py
import numpy as np
import torch.utils.data as data
from pydub import AudioSegment
import os
from os import walk, path, makedirs

rootdir="../KL_Study_HDF5_for_learning/data/"
sound_dir="./mp3/"

def extract_sound(sound_data,start_time, end_time,output_path):
    song = AudioSegment.from_mp3(sound_data)
    print("start time is",start_time)
    print("end_time is ",end_time)
    extract = song[startTime:endTime]
    extract.export(output_path, format="mp3")

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        filepath = subdir + os.sep + file
        if filepath.endswith("P1_6_min_walking_test.h5"):
          with h5py.File(filepath, 'r') as f:
            joint_angles=f.get('jointAngles')
            labels = f.get('labels')

            left_leg_sequences = np.asarray(labels.get('turn_segmentIdxs_left'))
            right_leg_sequences = np.asarray(labels.get('turn_segmentIdxs_right'))
            #Make Directory to save sound corresponding to each sequence
            output_dir_path= path.dirname(filepath)+"/sound_label/"+file.replace(".h5","")
            print(path.dirname(output_dir_path))
            if not path.exists(path.dirname(output_dir_path)):
              makedirs(path.dirname(output_dir_path)) 
            
            for i in range(len(left_leg_sequences)):
              sequence = left_leg_sequences[i]
              print(sequence)
              start_index=sequence[0]
              end_index=sequence[1]
              startval=start_index/60
              print("start value is",startval)
              startMin = int(startval/60)
              startSec = startval%60
              endval=end_index/60
              print("End value is",endval)
              endMin = int(endval/60)
              endSec = endval%60
              # Time to miliseconds
              startTime = startMin*60*1000+startSec*1000
              endTime = endMin*60*1000+endSec*1000
              output_path=output_dir_path+"_leftseq_"+str(i)+".mp3"
              print(output_path)
              sound_data=sound_dir+"transformed_"+file.replace(".h5",".mp3")
              print(sound_data)
              extract_sound(sound_data,startTime,endTime,output_path)
              
            
            for i in range(len(right_leg_sequences)):
              sequence = right_leg_sequences[i]
              start_index=sequence[0]
              end_index=sequence[1]

              startval=start_index/60
              print("start value is",startval)
              startMin = int(startval/60)
              startSec = startval%60
              endval=end_index/60
              print("End value is",endval)
              endMin = int(endval/60)
              endSec = endval%60
              #print("Left start min,startsec and end min end sec is",startMin,startSec,endMin,endSec)
              # Time to miliseconds
              startTime = startMin*60*1000+startSec*1000
              endTime = endMin*60*1000+endSec*1000
              output_path=output_dir_path+"_rightseq_"+str(i)+".mp3"
              print(output_path)
              sound_data=sound_dir+"transformed_"+file.replace(".h5",".mp3")
              extract_sound(sound_data,startTime,endTime,output_path)
              
              

          