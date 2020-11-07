import torch
import torch.utils.data as data
import torch.nn as nn



class Encoder(nn.Module):
    def __init__(self, seq_len, num_features, embedding_dim):
        super(Encoder, self).__init__()

        self.seq_len, self.num_features = seq_len, num_features
        self.hidden1 , self.hidden2 = 2 * embedding_dim, 4 * embedding_dim
        self.latent_dim = embedding_dim
        self.num_layers = 1

        self.enc = torch.nn.LSTM(self.num_features, self.latent_dim, self.num_layers)
        self.relu = nn.ReLU()

    def forward(self, x):
        
        x = x.reshape((1, self.seq_len, self.num_features))
              
        x, (hidden_n, cell_n) = self.enc(x)
        #print('x shape ',x.shape)
        #print('hidden shape ', hidden_n.shape)
        #print('Latent shape ', self.latent_dim)
        encoded_input = self.relu(x)
        #print('x shape', encoded_input.shape)
        return encoded_input
        #return hidden_n.reshape((1, self.latent_dim, -1))

class Decoder(nn.Module):
    def __init__(self, seq_len, embedded_dim, output_dim=6):
        super(Decoder, self).__init__()

        self.seq_len= seq_len
        self.input_dim, self.output_dim = embedded_dim, output_dim
        self.hidden_dim= embedded_dim
        self.num_layers=1
        
        self.dec =  torch.nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        self.sigmoid = nn.Sigmoid()
        self.output_layer = nn.Linear(self.hidden_dim, output_dim)
        
        
    def forward(self, x):
        
        #x = x.repeat(self.seq_len, 1)
        x = x.reshape((1, self.seq_len, self.input_dim))

        x, (hidden_n, cell_n) = self.dec(x)
        x = self.sigmoid(x)
        x = x.reshape((self.seq_len, self.hidden_dim))
        #print(x.shape)
        return self.output_layer(x)



class Autoencoder(nn.Module):
    def __init__(self, seq_len, num_features, embedding_dim):
        super(Autoencoder, self).__init__()

        self.seq_len, self.num_features = seq_len, num_features
        self.embedding_dim = embedding_dim
        

        self.encoder = Encoder(seq_len, num_features, embedding_dim)
        self.decoder = Decoder(seq_len, embedding_dim, num_features)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
    
