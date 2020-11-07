import torch
import torch.utils.data as data
from preprocess import get_data_dimensions
from Convolutional_AE import ConvAutoencoder
from statistics import mean
import numpy as np
import torch.nn as nn


def train_model(model, dataset, lr, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # specify loss function
    criterion = nn.MSELoss()
    # specify optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_epochs = epochs
    
    for epoch in range(1, n_epochs+1):
        # monitor training loss
        losses, seq_pred = [], []
        ###################
        # train the model #
        ###################
        for data_item in dataset:
            data_item =data_item.reshape((len(data_item),1,1))
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(data_item.float())
            #print(outputs)
            # calculate the loss
            loss = criterion(outputs, data_item.float())
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            #perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            losses.append(loss.item())
            seq_pred.append(outputs)
            
       
        print("Epoch: {}, Loss: {:.4f}".format(str(epoch),(mean(losses))))
    
    return seq_pred,mean(losses)


def flatten(t):
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t
 
def data_to_1d(data_set):
    #to convert data into single dimension vector
    t_list=[]
    len_data=len(data_set)
    for i in range(len_data):
        v=flatten(data_set[i])
        t=v.reshape(len(v),1)
        t_list.append(t)
    return t_list
    
def encoding_1d(train_dataset,lr,epoch,logging=False):
    #returns model encoder, decoder and embeddings with loss
    train_set, seq_len, num_features = get_data_dimensions(train_dataset)
    train_1d= data_to_1d(train_set)
    train_data_1d,seq,num_of_features=get_data_dimensions(train_1d)
    data_1d = np.stack(train_data_1d)
    d_set=torch.from_numpy(data_1d)
    model = ConvAutoencoder()
    embeddings, f_loss = train_model(model, d_set, lr, epoch )

    return model.encoder, model.decoder, embeddings, f_loss







