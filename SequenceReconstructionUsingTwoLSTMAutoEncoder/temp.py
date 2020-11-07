import h5py
import math
import numpy as np
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import torch.utils.data as data
from os import walk

'''
Visualizing the distribution of sequences based on their length
'''

root_dir = '../../KL_Study_HDF5_for_learning/data/'
cnt = 0

sequences = []
max_len = 0
min_len = math.inf

lower_bound = 55
upper_bound = 85

for (dirpath, _, filenames) in walk(root_dir):
    cnt += 1
    if len(filenames) > 0:
        for f in filenames:
            if f.split('.')[-1] == 'h5':
                file_path = dirpath +'/'+ f
                print(file_path)
                with h5py.File(file_path, 'r') as f:
                    joint_angles=f.get('jointAngles')
                    labels = f.get('labels')

                    flexion_angles = np.asarray(joint_angles.get('Flexion_angles'))
                    left_turn_segments = np.asarray(labels.get('turn_segments_left'))
                    right_turn_segments = np.asarray(labels.get('turn_segments_right'))
                    initial_contact_left = np.asarray(labels.get('initial_contact_left'))
                    initial_contact_right = np.asarray(labels.get('initial_contact_right'))

                mean = flexion_angles.mean(axis=0)

                left_contacts = []
                temp_id = 0
                for idx, cl in enumerate(initial_contact_left):
                    if cl == 1:
                        if idx != temp_id+1:                  
                            left_contacts.append(idx)
                        temp_id = idx

                
                right_contacts = []
                temp_id = 0
                for idx, cr in enumerate(initial_contact_right):
                    if cr == 1:
                        if idx != temp_id+1:
                            right_contacts.append(idx)
                        temp_id = idx

                if left_contacts[0] < right_contacts[0]:
                    for i in range(1, len(left_contacts)):
                        if left_turn_segments[left_contacts[i-1]] != 1: 
                            seq = flexion_angles[left_contacts[i-1]:left_contacts[i]]
                            seq_len = len(seq)
                            if(seq_len <= upper_bound and seq_len >= lower_bound):
                                if upper_bound-seq_len > 0:
                                    repeat_tensor = np.tile(mean, (upper_bound-seq_len, 1))
                                    seq = np.append(seq, repeat_tensor, 0)
                                sequences.append(seq)
                else:
                    for i in range(1, len(right_contacts)):
                        if right_turn_segments[right_contacts[i-1]] != 1:
                            seq = flexion_angles[right_contacts[i-1]:right_contacts[i]]
                            seq_len = len(seq)
                            if(seq_len <= upper_bound and seq_len >= lower_bound):
                                if upper_bound-seq_len > 0:
                                    repeat_tensor = np.tile(mean, (upper_bound-seq_len, 1))
                                    seq = np.append(seq, repeat_tensor, 0)
                                sequences.append(seq)
                break

sequence_lengths = []

for seq in sequences:
    sequence_lengths.append(len(seq))
    if len(seq) > max_len:
        max_len = len(seq)
    if len(seq) < min_len:
        min_len = len(seq)

print('max len ', max_len)
print('min len ', min_len)
print(len(sequences))

mode = max(set(sequence_lengths), key=sequence_lengths.count)
print('mode ', mode)

num_bins = 20
arr = plt.hist(sequence_lengths, num_bins, facecolor='blue', alpha=0.5)
for i in range(num_bins):
    plt.text(arr[1][i],arr[0][i],str(arr[0][i]))

plt.xlabel('Sequence Length')
plt.ylabel('Sequence Count')
plt.show()