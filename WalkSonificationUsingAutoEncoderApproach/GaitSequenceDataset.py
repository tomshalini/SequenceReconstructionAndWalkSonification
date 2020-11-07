import os
import sys
import h5py
import torch
import pretty_midi
import numpy as np
import torch.utils.data as data
from os import walk
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class GaitSequenceDataset(data.Dataset):
    def __init__(self, root_dir, longest_sequence, shortest_sequence):
        self.root_dir = root_dir
        self.longest_sequence = longest_sequence
        self.shortest_sequence = shortest_sequence
        self.sequences = []
        self.frames = []
        self.file_names = []
        self.sequence_lengths = []
        self.piano_rolls = []
        self.leg = []
        self._load_data()

    def pad_piano_roll(self, piano_roll, max_length=120, pad_value=0):
        
        original_piano_roll_length = piano_roll.shape[1]

        padded_piano_roll = np.zeros((128, max_length))
        padded_piano_roll[:] = pad_value
        
        padded_piano_roll[: , -original_piano_roll_length:] = piano_roll

        return padded_piano_roll

    def _load_data(self):

        for i in range(1, 2):
            folder_name = 'P'+str(i)
            hdf_file_path = os.path.join(self.root_dir, folder_name, folder_name+'_6_min_walking_test.h5')
            midi_file_path_left_base = os.path.join(self.root_dir, folder_name, 'midi_files', folder_name+'_6_min_walking_test_leftseq_')
            midi_file_path_right_base = os.path.join(self.root_dir, folder_name, 'midi_files', folder_name+'_6_min_walking_test_rightseq_')
            if os.path.exists(hdf_file_path):
                with h5py.File(hdf_file_path, 'r') as f:

                    joint_angles=f.get('jointAngles')
                    labels = f.get('labels')
                    flexion_angles = np.asarray(joint_angles.get('Flexion_angles'))
                    left_turn_segments = np.asarray(labels.get('turn_segments_left'))
                    right_turn_segments = np.asarray(labels.get('turn_segments_right'))
                    left_leg_sequences = np.asarray(labels.get('turn_segmentIdxs_left'))
                    right_leg_sequences = np.asarray(labels.get('turn_segmentIdxs_right'))
                
                for i in range(1, len(left_leg_sequences)):
                    sequence = left_leg_sequences[i]
                    if left_turn_segments[sequence[0]] != 1:
                        seq = flexion_angles[sequence[0]:sequence[1]]
                        seq_len = len(seq)
                        if seq_len >= self.shortest_sequence and seq_len <= self.longest_sequence:
                            midi_path = midi_file_path_left_base+str(i)+'.mp3.mid'
                            if os.path.exists(midi_path):
                                # Read midi and load the piano roll into memory
                                pm = pretty_midi.PrettyMIDI(midi_path)
                                piano_roll = pm.get_piano_roll(60).transpose()
                                piano_roll[piano_roll > 0] = 1
                                if piano_roll.shape[0] > seq_len:
                                    piano_roll = piano_roll[:-(piano_roll.shape[0]-seq_len), :]
                                piano_roll = self.pad_piano_roll(piano_roll.transpose(), self.longest_sequence)
                                self.piano_rolls.append(piano_roll)

                                # Read the sequence
                                pad = np.array([[0, 0, 0, 0, 0, 0]]*(self.longest_sequence-seq_len))
                                if len(pad) > 0:
                                    seq = np.append(seq, pad, 0)
                                self.sequences.append(seq)
                                self.sequence_lengths.append(seq_len)
                                self.file_names.append(hdf_file_path.split('/')[-1])
                                self.frames.append(i)
                                self.leg.append('left')

                for i in range(1, len(right_leg_sequences)):
                    sequence = right_leg_sequences[i]
                    if right_turn_segments[sequence[0]] != 1:
                        seq = flexion_angles[sequence[0]:sequence[1]]
                        seq_len = len(seq)
                        if seq_len >= self.shortest_sequence and seq_len <= self.longest_sequence:
                            midi_path = midi_file_path_right_base+str(i)+'.mp3.mid'
                            if os.path.exists(midi_path):
                                # Read midi and load the piano roll into memory
                                pm = pretty_midi.PrettyMIDI(midi_path)
                                piano_roll = pm.get_piano_roll(60).transpose()
                                piano_roll[piano_roll > 0] = 1
                                if piano_roll.shape[0] > seq_len:
                                    piano_roll = piano_roll[:-(piano_roll.shape[0]-seq_len), :]
                                piano_roll = self.pad_piano_roll(piano_roll.transpose(), self.longest_sequence)
                                self.piano_rolls.append(piano_roll)

                                # Read the sequence
                                pad = np.array([[0, 0, 0, 0, 0, 0]]*(self.longest_sequence-seq_len))
                                if len(pad) > 0:
                                    seq = np.append(seq, pad, 0)
                                self.sequences.append(seq)
                                self.sequence_lengths.append(seq_len)
                                self.file_names.append(hdf_file_path.split('/')[-1])
                                self.frames.append(i)
                                self.leg.append('right')


    def load_data(self):
        for (dirpath, _, filenames) in walk(self.root_dir):
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
                            left_leg_sequences = np.asarray(labels.get('turn_segmentIdxs_left'))
                            right_leg_sequences = np.asarray(labels.get('turn_segmentIdxs_right'))

                        for i in range(1, len(left_leg_sequences)):
                            sequence = left_leg_sequences[i]
                            if left_turn_segments[sequence[0]] != 1:
                                seq = flexion_angles[sequence[0]:sequence[1]]
                                seq_len = len(seq)
                                if seq_len >= self.shortest_sequence and seq_len <= self.longest_sequence:
                                    pad = np.array([[0, 0, 0, 0, 0, 0]]*(120-seq_len))
                                    if len(pad) > 0:
                                        seq = np.append(seq, pad, 0)
                                    self.sequences.append(seq)
                                    self.sequence_lengths.append(seq_len)
                                    self.file_names.append(file_path.split('/')[-1])
                                    self.frames.append(sequence)

                        for i in range(1, len(right_leg_sequences)):
                            sequence = right_leg_sequences[i]
                            if right_turn_segments[sequence[0]] != 1:
                                seq = flexion_angles[sequence[0]:sequence[1]]
                                seq_len = len(seq)
                                if seq_len >= self.shortest_sequence and seq_len <= self.longest_sequence:
                                    pad = np.array([[0, 0, 0, 0, 0, 0]]*(120-seq_len))
                                    if len(pad) > 0:
                                        seq = np.append(seq, pad, 0)
                                    self.sequences.append(seq)
                                    self.sequence_lengths.append(seq_len)
                                    self.file_names.append(file_path.split('/')[-1])
                                    self.frames.append(sequence)

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):

        sample = {'sequence' : torch.FloatTensor(self.sequences[idx]),
                    'piano_roll' : torch.FloatTensor(self.piano_rolls[idx]),
                    'sequence_length' : self.sequence_lengths[idx],
                    'file_name' : self.file_names[idx],
                    'frame' : self.frames[idx],
                    'leg' : self.leg[idx]}
        return sample

'''        sample = {'sequence' : torch.FloatTensor(self.sequences[idx]),
                    'sequence_length' : self.sequence_lengths[idx],
                    'file_name' : self.file_names[idx],
                    'frame' : self.frames[idx]}
'''

# data_dir = '../../KL_Study_HDF5_for_learning/data/'
# dataset = GaitSequenceDataset(root_dir = data_dir,
#                                     longest_sequence = 120,
#                                     shortest_sequence = 55)

# print(dataset[0])
