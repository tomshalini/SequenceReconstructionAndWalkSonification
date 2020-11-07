from preprocess import prepare_dataset
import torch.utils.data as data
from train_conv_AE import encoding_1d
from GaitSequenceDataset import GaitSequenceDataset
import torch

data_dir = '../../KL_Study_HDF5_for_learning/data/'

def main():
    batch_size = 1
    lr=1e-3
    epochs=10

    dataset = GaitSequenceDataset(root_dir = data_dir,
                                longest_sequence = 120,
                                shortest_sequence = 55)

    dataloader = data.DataLoader(dataset, batch_size, shuffle=True)
    #divide data into train and test
    train_dataset,test_dataset=prepare_dataset(dataloader)
    #embedding and loss of dataset
    encoder, decoder, embeddings, f_loss = encoding_1d(train_dataset,lr=lr,epoch=epochs)

    torch.save([encoder,decoder],'autoencoder_final.pkl')
    print(f_loss)

    #test_set, seq_len, num_features = get_data_dimensions(test_dataset[0:1])
    test_encoding = encoder(test_dataset[0:1].float())
    test_decoding = decoder(test_encoding)
    

if __name__ == "__main__":
    main()