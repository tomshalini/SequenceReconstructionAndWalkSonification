import sys
import math
import h5py
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from model import RNN
from trainer import train, validate
from torch.utils.tensorboard import SummaryWriter
from GaitSequenceDataset import GaitSequenceDataset
from torch.utils.data.sampler import SubsetRandomSampler
from utils import save_checkpoint, adjust_learning_rate, sonify_sequence

mode = 'test'
batch_size = 1
num_features = 6
num_classes = 128
sequence_length = 85
hidden_dimension = 128

num_epochs = 200
clip_gradient = 1
learning_rate = 1e-4
load_pretrained = True
loss_display_interval = 100

checkpoint_path = './result/BEST_checkpoint.pth'

data_dir = '../KL_Study_HDF5_for_learning/data/'

device = torch.device("cuda")

def generate_train_validation_samplers(dataset, validation_split):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(dataset_size * validation_split))
    np.random.shuffle(indices)
    train_indices, validation_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(validation_indices)

    return train_sampler, validation_sampler

'''
dataset = GaitSequenceDataset(root_dir = data_dir,
                                    longest_sequence = 120,
                                    shortest_sequence = 55)
train_sampler, validation_sampler = generate_train_validation_samplers(dataset, validation_split=0.2)

print('Building dataloaders..')
train_dataloader = data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
validation_dataloader = data.DataLoader(dataset, batch_size=1, sampler=validation_sampler, drop_last=True)
'''
'''
model = RNN(num_features, hidden_dimension, num_classes, num_layers = 2)
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
'''
file_path = '../KL_Study_HDF5_for_learning/data/P1/P1_6_min_walking_test.h5'


with h5py.File(file_path, 'r') as f:
    joint_angles=f.get('jointAngles')
    labels = f.get('labels')

    sequence = torch.FloatTensor(joint_angles.get('Flexion_angles')).unsqueeze(1).to(device)

sequence_length = torch.tensor([sequence.shape[0]])

print(sequence.shape)
model = RNN(num_features, hidden_dimension, num_classes, num_layers = 2)
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
sonify_sequence(model, sequence, sequence_length)

print('Done...')

'''
print('epocs',checkpoint['epoch'])
print('epochs since improvement',checkpoint['epochs_since_improvement'])
print(model)
for batch_idx, val_data in enumerate(validation_dataloader):
    sequence = val_data['sequence'].permute(1, 0, 2).to(device)
    piano_roll = val_data['piano_roll'].permute(1, 0, 2).to(device)
    sequence_length = val_data['sequence_length']
    file_name = val_data['file_name']
    frame = val_data['frame']
    leg = val_data['leg']
    sonify_sequence(model, sequence, sequence_length)
    print(file_name, frame, leg)
    break
'''