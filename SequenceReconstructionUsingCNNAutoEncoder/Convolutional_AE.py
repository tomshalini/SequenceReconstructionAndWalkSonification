import torch.nn.functional as F
import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 1 --> 128)
        self.conv1 = nn.Conv1d(1, 128, 1)      
        self.conv2 = nn.Conv1d(128, 64, 1)        
        self.conv3 = nn.Conv1d(64, 32, 1)
        self.pool = nn.MaxPool1d(1)

        
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv = nn.ConvTranspose1d(32, 64, 1)
        self.t_conv1 = nn.ConvTranspose1d(64, 128, 1)
        self.t_conv2 = nn.ConvTranspose1d(128, 1, 1)


    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)  
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
       
        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv(x))
        x = F.relu(self.t_conv1(x))
        x = torch.sigmoid(self.t_conv2(x))
        
        return x
    
#model = ConvAutoencoder()
#print(model)






