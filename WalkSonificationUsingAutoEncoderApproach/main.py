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

mode = 'train'
batch_size = 12
num_features = 6
sequence_length = 85
num_classes = 128
hidden_dimension = 512
embedding_dimension = 256

num_epochs = 200
clip_gradient = 1
learning_rate = 1e-4
load_pretrained = False
loss_display_interval = 20

checkpoint_path = './result/BEST_checkpoint.pth.tar'
# checkpoint_path = './result/checkpoint.pth'

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
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
    validation_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=validation_sampler, drop_last=True)

    if load_pretrained is True:
        print('Loading pretrained model..')
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        encoder = checkpoint['encoder']
        decoder = checkpoint['decoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        decoder_optimizer = checkpoint['decoder_optimizer']

    else:
        print('Creating model..')
        encoder = Encoder(sequence_length, num_features, embedding_dimension)
        decoder = Decoder(embedding_dimension, num_classes, hidden_dimension, sequence_length)
        encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

    criterion = nn.MSELoss().to(device)

    if mode == 'train':

        summary = SummaryWriter()
        #summary = None

        encoder.to(device)
        decoder.to(device)

        for epoch in range(start_epoch, start_epoch+num_epochs):

            if epochs_since_improvement == 20 :
                break

            if epochs_since_improvement > 0 and epochs_since_improvement % 4 == 0:
                adjust_learning_rate(encoder_optimizer, 0.8)

            train(encoder, decoder, train_dataloader, encoder_optimizer, decoder_optimizer, criterion, 
                    clip_gradient, device, epoch, num_epochs, summary, loss_display_interval)

            current_loss = validate(encoder, decoder, validation_dataloader, criterion, device, epoch, num_epochs, 
                                summary, loss_display_interval)

            is_best = max_loss > current_loss
            max_loss = min(max_loss, current_loss)
            if not is_best:
                epochs_since_improvement += 1
                print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
            else:
                epochs_since_improvement = 0

            save_checkpoint(epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer, is_best)

    else:
        print('testing...')
        encoder.to(device)
        decoder.to(device)
        encoder.eval()
        decoder.eval()
        for batch_idx, data in enumerate(validation_dataloader):
            sequence = data['sequence'][0].unsqueeze(0).permute(1, 0, 2).to(device)
            seq_len = data['sequence_length'][0]
            x ,(hidden_state, cell_state)= encoder(sequence)
            prediction = decoder(hidden_state)
            
            sequence = sequence.squeeze(1).detach().cpu().numpy()
            prediction = prediction.squeeze(1).detach().cpu().numpy()

            print(sequence.shape)
            hip_angles_gt = sequence[:seq_len, [0,3]]
            knee_angles_gt = sequence[:seq_len, [1,4]]
            ankle_angles_gt = sequence[:seq_len, [2,5]]

            hip_angles_pred = prediction[:seq_len, [0,3]]
            knee_angles_pred = prediction[:seq_len, [1,4]]
            ankle_angles_pred = prediction[:seq_len, [2,5]]

            time = np.arange(0, len(hip_angles_gt), 1)
            
            fig, axs = plt.subplots(4)
            # fig.suptitle('Hip angle reconstruction')
            # axs[0].plot(time, hip_angles_gt[:,0])
            # axs[0].set_title('Left hip ground truth')
            # axs[1].plot(time, hip_angles_pred[:,0])
            # axs[1].set_title('Left hip prediction')
            # axs[2].plot(time, hip_angles_gt[:,1])
            # axs[2].set_title('Right hip ground truth')
            # axs[3].plot(time, hip_angles_pred[:,1])
            # axs[3].set_title('Right hip prediction')

            # fig.suptitle('Knee angle reconstruction')
            # axs[0].plot(time, knee_angles_gt[:,0])
            # axs[0].set_title('Left knee ground truth')
            # axs[1].plot(time, knee_angles_pred[:,0])
            # axs[1].set_title('Left knee prediction')
            # axs[2].plot(time, knee_angles_gt[:,1])
            # axs[2].set_title('Right knee ground truth')
            # axs[3].plot(time, knee_angles_pred[:,1])
            # axs[3].set_title('Right knee prediction')

            fig.suptitle('Ankle angle reconstruction')
            axs[0].plot(time, ankle_angles_gt[:,0])
            axs[0].set_title('Left ankle ground truth')
            axs[1].plot(time, ankle_angles_pred[:,0])
            axs[1].set_title('Left ankle prediction')
            axs[2].plot(time, ankle_angles_gt[:,1])
            axs[2].set_title('Right ankle ground truth')
            axs[3].plot(time, ankle_angles_pred[:,1])
            axs[3].set_title('Right ankle prediction')

            plt.show()

            break



if __name__ == '__main__':
    main()