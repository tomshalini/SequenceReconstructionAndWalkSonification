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

# Training Parameters
mode = 'train'
batch_size = 8
num_features = 6
sequence_length = 85
hidden_dimension = 512
embedding_dimension = 512

num_epochs = 200
clip_gradient = 1
learning_rate = 1e-4
load_pretrained = False
loss_display_interval = 20

# Base name = optimizer + embedding_dimension
base_name = 'RMSprop_512_'

# Checkpoint paths to save the latest and the best models
best_checkpoint_path = './result/BEST_checkpoint.pth.tar'
checkpoint_path = './result/checkpoint.pth'

# Path tp data directory
data_dir = '../../KL_Study_HDF5_for_learning/data/'

# If cuda is not avaible use torch.device("cpu")
device = torch.device("cuda")

# Sampler function to split the data into training and validation
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

    # Creating custom dataset
    dataset = GaitSequenceDataset(root_dir = data_dir,
                                    longest_sequence = 85,
                                    shortest_sequence = 55)

    # Saplers for training and validation dataloaders
    train_sampler, validation_sampler = generate_train_validation_samplers(dataset, validation_split=0.2)

    print('Building dataloaders..')
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
    validation_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=validation_sampler, drop_last=True)

    # Loading a pretrained model
    if load_pretrained is True:
        print('Loading pretrained model..')
        checkpoint = torch.load(best_checkpoint_path)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        encoder = checkpoint['encoder']
        decoder = checkpoint['decoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        decoder_optimizer = checkpoint['decoder_optimizer']

    else:
        print('Creating model..')
        encoder = Encoder(sequence_length, num_features, embedding_dimension)
        decoder = Decoder(embedding_dimension, num_features, hidden_dimension, sequence_length)
        encoder_optimizer = torch.optim.RMSprop(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = torch.optim.RMSprop(decoder.parameters(), lr=learning_rate)

    # Mean Squared Loss
    criterion = nn.MSELoss().to(device)

    if mode == 'train':

        # Using summary writer for logging
        summary = SummaryWriter()

        encoder.to(device)
        decoder.to(device)

        for epoch in range(start_epoch, start_epoch+num_epochs):

            # Early stopping if the model does not learn for consecutive 10 epochs
            if epochs_since_improvement == 10 :
                break

            # Lower the learning rate by 0.2 after every 4th epoch with no learning
            if epochs_since_improvement > 0 and epochs_since_improvement % 4 == 0:
                adjust_learning_rate(encoder_optimizer, 0.8)

            # Train
            train(encoder, decoder, train_dataloader, encoder_optimizer, decoder_optimizer, criterion, 
                    clip_gradient, device, epoch, num_epochs, summary, loss_display_interval)

            # Validate
            current_loss = validate(encoder, decoder, validation_dataloader, criterion, device, epoch, num_epochs, 
                                summary, loss_display_interval)

            is_best = max_loss > current_loss
            max_loss = min(max_loss, current_loss)
            if not is_best:
                epochs_since_improvement += 1
                print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
            else:
                epochs_since_improvement = 0

            save_checkpoint(epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer, is_best, current_loss, base_name)

    else:
        # This part is used for the prurpose of visualizations.
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
            
            # fig, axs = plt.subplots(2)
            # fig.suptitle('Hip angle reconstruction')
            # axs[0].plot(time, hip_angles_gt[:,0])
            # axs[0].set_title('Left hip ground truth')
            # axs[1].plot(time, hip_angles_pred[:,0])
            # axs[1].set_title('Left hip prediction')

            plt.plot(time, ankle_angles_gt[:,1], label='Ground truth')
            plt.plot(time, ankle_angles_pred[:,1], label='Prediction')
            plt.title('Right-ankle angle reconstruction')
            plt.legend()

            # axs[0].plot(time, hip_angles_gt[:,1])
            # axs[0].set_title('Right hip ground truth')
            # axs[1].plot(time, hip_angles_pred[:,1])
            # axs[1].set_title('Right hip prediction')

            # fig.suptitle('Knee angle reconstruction')
            # axs[0].plot(time, knee_angles_gt[:,0])
            # axs[0].set_title('Left knee ground truth')
            # axs[1].plot(time, knee_angles_pred[:,0])
            # axs[1].set_title('Left knee prediction')
            # axs[0].plot(time, knee_angles_gt[:,1])
            # axs[0].set_title('Right knee ground truth')
            # axs[1].plot(time, knee_angles_pred[:,1])
            # axs[1].set_title('Right knee prediction')

            # fig.suptitle('Ankle angle reconstruction')
            # axs[0].plot(time, ankle_angles_gt[:,0])
            # axs[0].set_title('Left ankle ground truth')
            # axs[1].plot(time, ankle_angles_pred[:,0])
            # axs[1].set_title('Left ankle prediction')
            # axs[0].plot(time, ankle_angles_gt[:,1])
            # axs[0].set_title('Right ankle ground truth')
            # axs[1].plot(time, ankle_angles_pred[:,1])
            # axs[1].set_title('Right ankle prediction')

            plt.show()

            break



if __name__ == '__main__':
    main()