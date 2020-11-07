import torch
import sys
import math
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from matplotlib import pyplot as plt
from model import RNN
from trainer import train, validate
from torch.utils.tensorboard import SummaryWriter
from GaitSequenceDataset import GaitSequenceDataset
from torch.utils.data.sampler import SubsetRandomSampler
from utils import save_checkpoint, adjust_learning_rate, sonify_sequence

mode = 'test'
batch_size = 4
num_features = 6
num_classes = 128
sequence_length = 85
hidden_dimension = 128

num_epochs = 200
clip_gradient = 1
learning_rate = 1e-4
load_pretrained = True
loss_display_interval = 100

checkpoint_path = './result/checkpoint.pth'

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

def main():

    start_epoch = 0
    max_loss = math.inf
    epochs_since_improvement = 0

    dataset = GaitSequenceDataset(root_dir = data_dir,
                                    longest_sequence = 85,
                                    shortest_sequence = 55)

    train_sampler, validation_sampler = generate_train_validation_samplers(dataset, validation_split=0.2)

    print('Building dataloaders..')
    train_dataloader = data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_dataloader = data.DataLoader(dataset, batch_size=1, sampler=validation_sampler, drop_last=True)

    model = RNN(num_features, hidden_dimension, num_classes, num_layers = 2).to(device)
    
    
    if load_pretrained is True:
        print('Loading pretrained model..')
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = checkpoint['optimizer']

    else:
        print('Creating model..')
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss().to(device)

    if mode == 'train':

        summary = SummaryWriter()
        #summary = None

        model.to(device)
        print('###########    ',model)
        
        for epoch in range(start_epoch, start_epoch+num_epochs):

            if epochs_since_improvement == 20 :
                break

            if epochs_since_improvement > 0 and epochs_since_improvement % 4 == 0:
                adjust_learning_rate(optimizer, 0.8)

            train(model, train_dataloader, optimizer, criterion, clip_gradient, device, epoch, num_epochs, summary, loss_display_interval)

            current_loss = validate(model, validation_dataloader, criterion, device, epoch, num_epochs, summary, loss_display_interval)

            is_best = max_loss > current_loss
            max_loss = min(max_loss, current_loss)
            if not is_best:
                epochs_since_improvement += 1
                print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
            else:
                epochs_since_improvement = 0

            save_checkpoint(epoch, epochs_since_improvement, model, optimizer, is_best)

            print('Current loss : ', current_loss, ' Max loss : ' ,max_loss)


    else:
        print('testing...')
        model = RNN(num_features, hidden_dimension, num_classes, num_layers = 2)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        print(model)
        for batch_idx, val_data in enumerate(validation_dataloader):
            sequence = val_data['sequence'].permute(1, 0, 2).to(device)
            piano_roll = val_data['piano_roll'].permute(1, 0, 2).squeeze(1).to('cpu')
            sequence_length = val_data['sequence_length']
            file_name = val_data['file_name']
            frame = val_data['frame']
            leg = val_data['leg']
            sonify_sequence(model, sequence, sequence_length)
            plt.imshow(piano_roll)
            plt.show()
            print(file_name, frame, leg)
            break

if __name__ == '__main__':
    main()