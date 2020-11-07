import sys
import torch
import torch.nn as nn
from  torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = torch.device("cuda")

class RNN(nn.Module):
    
    def __init__(self, num_features, hidden_dimension, num_classes, num_layers = 2):
        
        super(RNN, self).__init__()

        self.num_features = num_features
        self.hidden_dimension = hidden_dimension
        self.num_classes = num_classes
        self.num_layers = num_layers
        
        self.lstmcell = nn.LSTMCell(num_features, hidden_dimension, bias=True)
        #for name, _ in self.lstmcell.named_parameters():
        #    print(name)
        self.bn = nn.BatchNorm1d(hidden_dimension)
        self.logits_fc = nn.Linear(hidden_dimension, num_classes)
    
    def forward(self, input_sequences, input_sequence_lengths):
        batch_size = input_sequences.shape[1]
        h = torch.zeros([batch_size, self.hidden_dimension]).to(device)
        c = torch.zeros([batch_size, self.hidden_dimension]).to(device)
        predictions = torch.zeros([input_sequences.shape[0], batch_size, self.num_classes], dtype=torch.float)
        for t in range(max(input_sequence_lengths).item()):
            batch_size_t = sum([l > t for l in input_sequence_lengths])
            h, c = self.lstmcell(torch.cuda.FloatTensor(input_sequences[t, : batch_size_t,:]), (h[:batch_size_t,:], c[:batch_size_t,:]))
            logits = self.logits_fc(h)
            predictions[t, :batch_size_t] = logits
        logits = predictions.transpose(0, 1).contiguous()
        neg_logits = (1 - logits)
        binary_logits = torch.stack((logits, neg_logits), dim=3).contiguous()
        logits_flatten = binary_logits.view(-1, 2)
        return logits_flatten            
