import sys
import torch
sys.path.append('./midi/')

from matplotlib import pyplot as plt
from midi_utils import midiread, midiwrite

def save_checkpoint(epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer, is_best, current_loss, base_name):
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
             'encoder': encoder,
             'decoder': decoder,
             'error' : current_loss,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = base_name+'checkpoint.pth.tar'
    torch.save(state, './result/'+filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, './result/BEST_' + filename)


def adjust_learning_rate(optimizer, shrink_factor):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print('New learning rate is : ', optimizer.param_groups[0]['lr'])

def sonify_sequence(model, sequence, sequence_length):

    h = torch.zeros([1, 512]) 
    c = torch.zeros([1, 512])
    predictions = torch.zeros([sequence_length[0], 1, 128], dtype=torch.float)
    for i in range(sequence_length):
        h, c = model.lstmcell(torch.FloatTensor(sequence[i]), (h,c))
        logits = model.logits_fc(h)
        predictions[i] = logits

    predictions = predictions.detach().squeeze(1).numpy()
    predictions[predictions>=0.03] = 1
    predictions[predictions<0.03] = 0

    midiwrite('midi_test.midi', predictions.transpose(), dt=0.025)