import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from os import walk, path, makedirs

#file_path = '../../KL_Study_HDF5_for_learning/data/P1/P1_6_min_walking_test.h5'
rootdir="../KL_Study_HDF5_for_learning/data/"
output_dir="./transformed_data/"
if not path.exists(path.dirname(output_dir)):
    makedirs(path.dirname(output_dir))

longest_sequence = 120
shortest_sequence = 55
sequences = []
sequence_lengths = []
frames = []

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        filepath = subdir + os.sep + file
        if filepath.endswith(".h5"):
            with h5py.File(filepath, 'r') as f:
                output_filename= output_dir+"transformed_"+file.replace(".h5",".csv")
                joint_angles=f.get('jointAngles')
                labels = f.get('labels')

                flexion_angles = np.asarray(joint_angles.get('Flexion_angles'))

                transformer = StandardScaler()
                transformer.fit(flexion_angles)
                scaled_data = transformer.transform(flexion_angles)
                
                transformed_data = scaled_data * 500

                transformed_data = [transformed_data[i] for i in range(0, len(transformed_data), 5)]  # try different values for skip [5, 6, 7...].
                np.savetxt(output_filename, transformed_data, delimiter=',')
                