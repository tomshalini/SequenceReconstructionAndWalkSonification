import h5py
import numpy as np
import torch.utils.data as data
from os import walk

class GaitSequenceDataset(data.Dataset):
    def __init__(self, root_dir, longest_sequence, shortest_sequence):
        self.root_dir = root_dir
        self.longest_sequence = longest_sequence
        self.shortest_sequence = shortest_sequence
        self.sequences = []
        self.load_data()

    def load_data(self):
        for (dirpath, _, filenames) in walk(self.root_dir):
            if len(filenames) > 0:
                for f in filenames:
                    if f.split('.')[-1] == 'h5':
                        file_path = dirpath +'/'+ f
                        # print(file_path)
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
                                    if(seq_len <= self.longest_sequence and seq_len >= self.shortest_sequence):
                                        if self.longest_sequence-seq_len > 0:
                                            repeat_tensor = np.tile(mean, (self.longest_sequence-seq_len, 1))
                                            seq = np.append(seq, repeat_tensor, 0)
                                        self.sequences.append(seq)
                        else:
                            for i in range(1, len(right_contacts)):
                                if right_turn_segments[right_contacts[i-1]] != 1:
                                    seq = flexion_angles[right_contacts[i-1]:right_contacts[i]]
                                    seq_len = len(seq)
                                    if(seq_len <= self.longest_sequence and seq_len >= self.shortest_sequence):
                                        if self.longest_sequence-seq_len > 0:
                                            repeat_tensor = np.tile(mean, (self.longest_sequence-seq_len, 1))
                                            seq = np.append(seq, repeat_tensor, 0)
                                        self.sequences.append(seq)
                        break


    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]