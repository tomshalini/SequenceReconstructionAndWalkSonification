import torch.utils.data as data
from GaitSequenceDataset import GaitSequenceDataset
from preprocess import prepare_dataset
from train_ae_single import encoding
import torch

data_dir = '../../KL_Study_HDF5_for_learning/data/'

def main():
    lr=1e-3
    epochs=10

    dataset = GaitSequenceDataset(root_dir =data_dir,
                                longest_sequence = 85,
                                shortest_sequence = 55)

    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True)
    train_dataset,test_dataset=prepare_dataset(dataloader)
    encoder, decoder, embeddings, f_loss = encoding(train_dataset,32,lr=lr,epoch=epochs,logging=True)
    torch.save([encoder,decoder],'autoencoder_final.pkl')
    print(f_loss)

    #test_set, seq_len, num_features = get_data_dimensions(test_dataset[0:1])
    test_encoding = encoder(test_dataset[0:1].float())
    test_decoding = decoder(test_encoding)

 

if __name__ == "__main__":
    main()