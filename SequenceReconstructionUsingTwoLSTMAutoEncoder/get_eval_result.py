import torch

checkpoint_path = './result/BEST_RMSprop_512_checkpoint.pth.tar'

checkpoint = torch.load(checkpoint_path)

print(checkpoint['error'])