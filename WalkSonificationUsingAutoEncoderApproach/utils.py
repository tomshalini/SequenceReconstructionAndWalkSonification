import sys
import torch
sys.path.append('./midi/')

from matplotlib import pyplot as plt
from midi_utils import midiread, midiwrite

def save_checkpoint(epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer, is_best):
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
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint.pth.tar'
    torch.save(state, './result/'+filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, './result/BEST_' + filename)


def adjust_learning_rate(optimizer, shrink_factor):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print('New learning rate is : ', optimizer.param_groups[0]['lr'])

def sonify_sequence(encoder, decoder, sequence, sequence_length, file_name=None):

    x ,(hidden_state, cell_state) = encoder(sequence)
    prediction = decoder(hidden_state)

    predictions = prediction.detach().squeeze(1).numpy()
    predictions=predictions*10000
    print(predictions)
    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[1]):
            if predictions[i][j]<40 or predictions[i][j]>80:
                predictions[i][j]=0
            elif predictions[i][j]>=40 or prediction[i][j]<=80:
                predictions[i][j]=1
    #predictions[predictions>=15]= 1
    #predictions[predictions<=16]= 1

    

    plt.imshow(predictions)
    plt.show()

    print(predictions.shape)
    print(predictions)

    if file_name is None:
        midiwrite('encoder_decoder.midi', predictions.transpose(), dt=0.025)
    else:
        midiwrite(file_name, predictions.transpose(), dt=0.025)