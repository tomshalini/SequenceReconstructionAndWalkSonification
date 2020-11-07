import h5py
import numpy as np

file_path = '../../KL_Study_HDF5_for_learning/data/P1/P1_6_min_walking_test.h5'

longest_sequence = 125
shortest_sequence = 55
sequences = []
sequence_lengths = []
frames = []

with h5py.File(file_path, 'r') as f:

    joint_angles=f.get('jointAngles')
    labels = f.get('labels')

    flexion_angles = np.asarray(joint_angles.get('Flexion_angles'))
    left_turn_segments = np.asarray(labels.get('turn_segments_left'))
    right_turn_segments = np.asarray(labels.get('turn_segments_right'))
    left_leg_sequences = np.asarray(labels.get('turn_segmentIdxs_left'))
    right_leg_sequences = np.asarray(labels.get('turn_segmentIdxs_right'))

    if left_leg_sequences[1][0] < right_leg_sequences[1][0]:
        # left leg first
        for i in range(1, len(left_leg_sequences)):
            sequence = left_leg_sequences[i]
            if left_turn_segments[sequence[0]] != 0:
                seq = flexion_angles[sequence[0]:sequence[1]]
                seq_len = len(seq)
                if seq_len >= shortest_sequence and seq_len <= longest_sequence:
                    sequences.append(seq)
                    sequence_lengths.append(seq_len)
                    frames.append(sequence)

    else:
        # right leg first
        for i in range(1, len(right_leg_sequences)):
            sequence = right_leg_sequences[i]
            if right_turn_segments[sequence[0]] != 0:
                seq = flexion_angles[sequence[0]:sequence[1]]
                seq_len = len(seq)
                if seq_len >= shortest_sequence and seq_len <= longest_sequence:
                    sequences.append(seq)
                    sequence_lengths.append(seq)
                    frames.append(sequence)


print(frames[0])

# print(len(frames))