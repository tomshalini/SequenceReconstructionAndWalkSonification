import sys
import math
import h5py
import torch
import numpy as np
import torch.nn as nn
from model import RNN
import torch.utils.data as data
from sklearn import preprocessing
from trainer import train, validate
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from GaitSequenceDataset import GaitSequenceDataset
from torch.utils.data.sampler import SubsetRandomSampler
from utils import save_checkpoint, adjust_learning_rate, sonify_sequence
from sklearn.metrics import confusion_matrix

mode = 'test'
batch_size = 1
num_features = 6
num_classes = 128
sequence_length = 85
hidden_dimension = 128

load_pretrained = False
loss_display_interval = 100

checkpoint_path = './result/BEST_checkpoint.pth'

data_dir = '../KL_Study_HDF5_for_learning/data/'

device = torch.device("cpu")


dataset = GaitSequenceDataset(root_dir = data_dir,
                                longest_sequence = 85,
                                shortest_sequence = 55)

print('Building dataloaders..')
eval_dataloader = data.DataLoader(dataset, batch_size=batch_size)

model = RNN(num_features, hidden_dimension, num_classes, num_layers = 2).to(device)
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])

true_positives = []
false_positives = []
temp = np.ones((85, 128))

model.eval()
true_positives = []
false_positives = []
true_negatives = []
false_negatives = []
for batch_idx, val_data in enumerate(eval_dataloader):
    gt_piano_roll = val_data['piano_roll'].permute(0, 2, 1).squeeze(0).detach().numpy()
    sequence = val_data['sequence'].permute(1, 0, 2)
    seq_len = val_data['sequence_length']
    # gt = gt_piano_roll.detach().numpy()

# -----------------------------------------------------------------------------------

    h = torch.zeros([1, 128])
    c = torch.zeros([1, 128])
    pred_piano_roll = torch.zeros([85, 1, 128], dtype=torch.float)

    for i in range(sequence_length):
        h, c = model.lstmcell(torch.FloatTensor(sequence[i]), (h,c))
        logits = model.logits_fc(h)
        pred_piano_roll[i] = logits

    pred_piano_roll = pred_piano_roll.squeeze(1)

    pred_temp = np.zeros((85, 128))

    for i in range(pred_piano_roll.shape[0]):
        for j in range(pred_piano_roll.shape[1]):
            if pred_piano_roll[i][j]<2.5 or pred_piano_roll[i][j]>3.5:
                pred_temp[i][j]=0
            elif pred_piano_roll[i][j]>=2.5 or pred_piano_roll[i][j]<=3.5:
                pred_temp[i][j]=1
    
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

'''true_positives = preprocessing.normalize([true_positives])
false_positives = preprocessing.normalize([false_positives])
true_negatives = preprocessing.normalize([true_negatives])
false_negatives = preprocessing.normalize([false_negatives])
'''

print('True Positives : ', sum(true_positives)/len(true_positives))
print('False Positives : ', sum(false_positives)/len(false_positives))
print('True Negatives : ', sum(true_negatives)/len(true_negatives))
print('False Negatives : ', sum(false_negatives)/len(false_negatives))