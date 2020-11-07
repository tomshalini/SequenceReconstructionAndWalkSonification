import sys
import torch
import numpy as np
sys.path.append('./midi/')

from matplotlib import pyplot as plt
from midi_utils import midiread, midiwrite

def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, is_best):
    """
    Saves model checkpoint.

    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param model: model
    :param optimizer: optimizer to update weights
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'model_state_dict': model.state_dict(),
             'optimizer': optimizer}
    filename = 'checkpoint.pth'
    torch.save(state, './result/'+filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, './result/BEST_' + filename)

def plot_histogram(predictions,savefile):
    predictions=np.concatenate([p.flatten() for p in predictions])
    min_value=np.min(predictions)
    max_value=np.max(predictions)
    plt.hist(predictions, range=(min_value,max_value))
    plt.savefig(savefile)
    plt.close()

def np_plot_histogram(predictions,savefile):
    predictions=np.concatenate([p.flatten() for p in predictions])
    min_value=np.min(predictions)
    max_value=np.max(predictions)
    y, x = np.histogram(predictions, bins=np.linspace(min_value, max_value, (max_value-min_value)/1))

    nbins = y.size

    plt.bar(x[:-1], y, width=x[1]-x[0], color='red', alpha=0.5)
    plt.hist(predictions, bins=nbins, alpha=0.5)
    plt.grid(True)
    plt.savefig(savefile)

def adjust_learning_rate(optimizer, shrink_factor):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print('New learning rate is : ', optimizer.param_groups[0]['lr'])

def sonify_sequence(model, sequence, sequence_length):

    print('Sonifying...')
    h = torch.zeros([1, 128]).cuda()
    c = torch.zeros([1, 128]).cuda()
    predictions = torch.zeros([sequence_length[0], 1, 128], dtype=torch.float)
    for i in range(sequence_length):
        h, c = model.lstmcell(torch.cuda.FloatTensor(sequence[i]), (h,c))
        logits = model.logits_fc(h)
        predictions[i] = logits

    print('predictions : ', predictions)
    print('predictions shape : ', predictions.shape)
    predictions = predictions.detach().squeeze(1).numpy().transpose()
    print('##############')
    #predictions=predictions*10000
    #print(predictions)
    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[1]):
            if predictions[i][j]<2.5 or predictions[i][j]>3.5:
                predictions[i][j]=0
            elif predictions[i][j]>=2.5 or prediction[i][j]<=3.5:
                predictions[i][j]=1
    '''#predictions=np.concatenate([p.flatten() for p in predictions])
    print('Generated Piano roll is:',predictions)
    predictions[predictions<8] = 0
    predictions[predictions>=8] = 1'''
    #plot_histogram(predictions,'pianoroll_test.png')
    
    plt.imshow(predictions)
    plt.show()

    print(predictions.shape)
    print(predictions)
    
    
    print('Updated Piano roll is:',predictions)
    midiwrite('seq_to_seq.midi', predictions.transpose(), dt=0.020)