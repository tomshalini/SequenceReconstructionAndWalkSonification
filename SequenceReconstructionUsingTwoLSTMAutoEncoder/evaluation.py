import sys
import math
import h5py
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils.data as data
from models import Encoder, Decoder
from trainer import train, validate
from torch.utils.tensorboard import SummaryWriter
from GaitSequenceDataset import GaitSequenceDataset
from torch.utils.data.sampler import SubsetRandomSampler
from utils import save_checkpoint, adjust_learning_rate, sonify_sequence

mode = 'test'
batch_size = 1
num_features = 6
sequence_length = 120
hidden_dimension = 512
embedding_dimension = 256

# This code shows the distribution of sequence reconstruction losses

checkpoint_path = './result/BEST_checkpoint.pth.tar'

data_dir = '../../KL_Study_HDF5_for_learning/data/'

device = torch.device("cuda")

def main():

    dataset = GaitSequenceDataset(root_dir = data_dir,
                                    longest_sequence = 120,
                                    shortest_sequence = 55)

    print('Building dataloader..')
    eval_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True)

    print('Loading pretrained model..')
    checkpoint = torch.load(checkpoint_path)
    encoder = checkpoint['encoder']
    decoder = checkpoint['decoder']

    criterion = nn.L1Loss(reduction='sum').to(device)

    encoder.eval()
    decoder.eval()

    num_batches = len(eval_dataloader)
    losses = []

    with torch.no_grad():
        for batch_idx, data in enumerate(eval_dataloader):
        
            sequences = data['sequence'].permute(1, 0, 2).to(device)

            x ,(hidden_state, cell_state)= encoder(sequences)
            prediction = decoder(hidden_state)

            loss = criterion(prediction, sequences)
            losses.append(loss.item())

            print(batch_idx, '/', num_batches)
            # sys.exit()
    print(len(losses))
    num_bins = 30
    arr = plt.hist(losses, num_bins, facecolor='blue', alpha=0.5)
    for i in range(num_bins):
        plt.text(arr[1][i],arr[0][i],str(arr[0][i]))

    plt.xlabel('Losses')
    plt.ylabel('Sequences')
    plt.show()

if __name__ == '__main__':
    main()