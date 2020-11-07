import sys
import math
import h5py
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from models import Encoder, Decoder
from sklearn import preprocessing
from trainer import train, validate
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from GaitSequenceDataset import GaitSequenceDataset
from torch.utils.data.sampler import SubsetRandomSampler
from utils import save_checkpoint, adjust_learning_rate, sonify_sequence
from sklearn.metrics import confusion_matrix

# Evaluation code to extimate the nimber of correctly predicted notes from the piano roll

mode = 'test'
batch_size = 1
num_features = 6
sequence_length = 85
num_classes = 128
hidden_dimension = 512
embedding_dimension = 256

load_pretrained = False
loss_display_interval = 100

checkpoint_path = './result/BEST_checkpoint.pth.tar'

data_dir = '../KL_Study_HDF5_for_learning/data/'

device = torch.device("cpu")


dataset = GaitSequenceDataset(root_dir = data_dir,
                                longest_sequence = 85,
                                shortest_sequence = 55)

print('Building dataloaders..')
eval_dataloader = data.DataLoader(dataset, batch_size=batch_size)

checkpoint_path = './result/BEST_checkpoint.pth.tar'
checkpoint = torch.load(checkpoint_path)
encoder = checkpoint['encoder'].to(device)
decoder = checkpoint['decoder'].to(device)

true_positives = []
false_positives = []
temp = np.ones((128, 85))

encoder.eval()
decoder.eval()
#model.eval()
true_positives = []
false_positives = []
true_negatives = []
false_negatives = []
for batch_idx, val_data in enumerate(eval_dataloader):
    gt_piano_roll = val_data['piano_roll'].squeeze(0)
    sequence = val_data['sequence'].permute(1, 0, 2)
    seq_len = val_data['sequence_length']
    gt = gt_piano_roll.detach().numpy()

# -----------------------------------------------------------------------------------

    h = torch.zeros([1, 128])
    c = torch.zeros([1, 128])

    x ,(hidden_state, cell_state) = encoder(sequence)
    predictions = decoder(hidden_state).squeeze(1)
    pred_temp = np.zeros((128, 85))

    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[1]):
            if predictions[i][j]<40 or predictions[i][j]>80:
                pred_temp[i][j] = 0
            elif predictions[i][j]>=40 or predictions[i][j]<=80:
                pred_temp[i][j] = 1
# -----------------------------------------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(pred_temp.reshape(1, -1)[0], gt_piano_roll.reshape(1, -1)[0]).ravel()
 
    true_positives.append(tp)
    true_negatives.append(tn)
    false_positives.append(fp)
    false_negatives.append(fn)

    

print(true_positives)
print(false_positives)
print(true_negatives)
print(false_negatives)

true_positives = preprocessing.normalize([true_positives])
false_positives = preprocessing.normalize([false_positives])
true_negatives = preprocessing.normalize([true_negatives])
false_negatives = preprocessing.normalize([false_negatives])


print('True Positives : ', sum(true_positives[0])/len(true_positives[0]))
print('False Positives : ', sum(false_positives[0])/len(false_positives[0]))
print('True Negatives : ', sum(true_negatives[0])/len(true_negatives[0]))
print('False Negatives : ', sum(false_negatives[0])/len(false_negatives[0]))