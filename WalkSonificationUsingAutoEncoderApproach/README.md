# Walk Sonification using AutoEncoder approach

This project aims to achieve walk sonification, ie. generating sound using walking motion data. 

GaitSequenceDataset is the custom dataset that will return only the valid walking sequences which are used for sonfication.

Ground truth for sonifcation  is prepared using ground_truth_preperation.py and extract_sound.py from the GroundTruthPreperation folder.

The Encoder and Decoder used to encode the walking sequence and generate sound from the encoding are written in models.py.

To run the code, execute main.py, 
'''
python main.py
'''

load_and_test.py is used to generate a midi file corresponding to the input sequence. 

Training parameters can be modified in main.py