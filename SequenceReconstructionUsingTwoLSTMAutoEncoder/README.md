# Sequence Reconstruction using Two LSTM's

This project aims to achieve walking sequence reconstruction using two LSTM's. The Encoder and Decoder are written in models.py. Encoder will encode the walking sequence into a latent space representation, using which the decoder will learn to reconstruct the input sequence.

GaitSequenceDataset is the custom dataset that will return only the valid walking sequences.

evaluation.py contains the code to show the distribution of sequence reconstruction losses.

To run the code, execute main.py, 
'''
python main.py
'''

Training parameters can be modified in main.py