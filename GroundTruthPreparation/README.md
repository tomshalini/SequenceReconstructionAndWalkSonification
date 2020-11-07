# Ground Truth Preparation

Step 1: Generate transformed data: Scale by 500 and take every 5th data value.
        output_dir="./transformed_data/"

        '''
        python .\ground_truth_Preparation.py

        '''
Step 2: Generate mp3 corresponding to each file present in "./transformed_data/" using twotone.io (https://app.twotone.io/) using only hip joint angles. Exported mp3 length : 6minutes 40 seconds.

Step 3 : Use Audacity to generate 6minute mp3 from generated mp3 in step2 and save these mp3 files in sound_dir="./mp3/".

Step 4: Generate sound corresponding to each sequence.
        output_dir_path= "../KL_Study_HDF5_for_learning/data/P1/sound_label/"
        left leg sequence file name example: P1_6_min_walking_test_leftseq_0.mp3
        right leg sequence file name example: P1_6_min_walking_test_rightseq_0.mp3

        '''
        python .\extract_sound.py
        
        '''

Step 5: Generate midi files for each mp3 using online tool (https://www.bearaudiotool.com/mp3-to-midi) and save exported midi files in data folder cooresponding to each person.
        for example midis corresponding to P1 will get saved in to  "../KL_Study_HDF5_for_learning/data/P1/midi_files".
    
Step 6: Get sequence and corresponding piano roll using dataloader file.
        
        '''
        python .\GaitSequenceDataset.py

        '''

       
        
        