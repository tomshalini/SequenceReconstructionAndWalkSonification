import torch
import torch.utils.data as data
import numpy as np


def prepare_dataset(dataloader):
    samples_count=len(dataloader)
    train_samples_count = int(0.8*samples_count)
    test_samples_count = int(0.2*samples_count)

    total_dataset= torch.empty(1,85, 6, dtype=torch.float64)
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
    
    #returns the dataset with the sequence length and number of features
    
    train_set = [dataset[i] for i in range(len(dataset))]
    shape = torch.stack(train_set).shape
    assert(len(shape) == 3)
    print(shape)
  
    return train_set, shape[1], shape[2]