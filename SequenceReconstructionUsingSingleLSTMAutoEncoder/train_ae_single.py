import torch
import torch.utils.data as data
from preprocess import get_data_dimensions
from single_LSTM import Autoencoder
from torch.nn import MSELoss
import torch.nn as nn
from statistics import mean

def train_model(model, dataset, lr, epochs, logging):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.09)
    # criterion = CrossEntropyLoss()
    criterion = RMSELoss()
    
    for epoch in range(1, epochs + 1):
        model.train()
        if not epoch % 50:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr * (0.993 ** epoch)
        
        losses, embeddings = [], []
        for seq_true in dataset:
            optimizer.zero_grad()

            # Forward pass
            seq_true=seq_true.float()
            seq_pred = model(seq_true)
            
            loss = criterion(seq_pred, seq_true)

            # Backward pass
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            embeddings.append(seq_pred)
            
        print("Epoch: {}, Loss: {}".format(str(epoch), str(mean(losses))))
       
    return embeddings, mean(losses)

#Root mean square error function
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))


def encoding(train_dataset,encoding_dim,lr,epoch,logging=False):
    train_set, seq_len, num_features = get_data_dimensions(train_dataset)
    print(seq_len, num_features)
    model = Autoencoder(seq_len, num_features, encoding_dim)
    embeddings, f_loss = train_model(model, train_set, lr, epoch, logging  )

    return model.encoder, model.decoder, embeddings, f_loss



