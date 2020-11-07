import sys
import math
import h5py
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from trainer import train, validate
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from GaitSequenceDataset import GaitSequenceDataset
from torch.utils.data.sampler import SubsetRandomSampler
from utils import save_checkpoint, adjust_learning_rate, sonify_sequence

mode = 'test'
batch_size = 1
num_features = 6
num_classes = 128
sequence_length = 120
hidden_dimension = 512

num_epochs = 200
clip_gradient = 1
learning_rate = 1e-4
load_pretrained = True
loss_display_interval = 100

checkpoint_path = './result/BEST_checkpoint.pth.tar'

data_dir = '../KL_Study_HDF5_for_learning/data/'

device = torch.device("cpu")

def generate_train_validation_samplers(dataset, validation_split):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(dataset_size * validation_split))
    np.random.shuffle(indices)
    train_indices, validation_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(validation_indices)

    return train_sampler, validation_sampler


dataset = GaitSequenceDataset(root_dir = data_dir,
                                    longest_sequence = 85,
                                    shortest_sequence = 55)
train_sampler, validation_sampler = generate_train_validation_samplers(dataset, validation_split=0.2)

print('Building dataloaders..')
train_dataloader = data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
validation_dataloader = data.DataLoader(dataset, batch_size=1, sampler=validation_sampler, drop_last=True)

checkpoint = torch.load(checkpoint_path)
encoder = checkpoint['encoder'].to(device)
decoder = checkpoint['decoder'].to(device)

for batch_idx, val_data in enumerate(validation_dataloader):
    sequence = val_data['sequence'].permute(1, 0, 2).to(device)
    piano_roll = val_data['piano_roll'].permute(1, 0, 2).squeeze(1).to(device)
    sequence_length = val_data['sequence_length']
    file_name = val_data['file_name']
    frame = val_data['frame']
    leg = val_data['leg']

    print(file_name, leg, frame)

    sonify_sequence(encoder, decoder, sequence, sequence_length)

    plt.imshow(piano_roll)
    plt.show()

    break

print('Done...')
