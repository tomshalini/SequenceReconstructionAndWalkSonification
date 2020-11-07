import torch
import numpy as np


def prepare_dataset(dataloader):
    #this function divides the dataset into train and test datasets randomly
    samples_count=len(dataloader)
    train_samples_count = int(0.8*samples_count)
    test_samples_count = int(0.2*samples_count)
    seq_len=120
    num_features=6
    total_dataset= torch.empty(1, seq_len, num_features, dtype=torch.float64)
    for  i,sequence in enumerate(dataloader):
        total_dataset=torch.cat((total_dataset,sequence),0)
    total_dataset=total_dataset[1:]
    shuffled_indices = np.arange(total_dataset.shape[0])
    np.random.shuffle(shuffled_indices)

    shuffled_inputs = total_dataset[shuffled_indices]
    print(shuffled_inputs.shape)
    train_dataset=shuffled_inputs[:train_samples_count]
    test_dataset=shuffled_inputs[train_samples_count+test_samples_count:]
  
    return train_dataset,test_dataset


def get_data_dimensions(dataset):
    #To obtain the dimensions of train set. It returns dataset, sequence length and number of features
    train_set = [dataset[i] for i in range(len(dataset))]
    shape = torch.stack(train_set).shape
    assert(len(shape) == 3)
    print(shape)
  
    return train_set, shape[1], shape[2]