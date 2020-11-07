# Walk Sonification using Sequence to Sequence approach

This project aims to achieve walk sonification, ie. generating sound using walking motion data. 

GaitSequenceDataset is the custom dataset that will return only the valid walking sequences which are used for sonfication.

Ground truth for sonifcation  is prepared using ground_truth_preperation.py and extract_sound.py from the GroundTruthPreperation folder.

The sequence to sequence LSTM model is written in models.py.

To run the code, execute main.py, 
'''
python main.py
'''

load_and_test.py is used to generate a midi file corresponding to the input sequence. 

Training parameters can be modified in main.py