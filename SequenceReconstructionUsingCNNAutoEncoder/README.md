# WalkingSonification - 1D convolutional Neural Network Autoencoder

Takes walking motion data which contains joints information like flexion angles between the hip and upper-leg, upper leg and lower-leg, and lower-leg and feet.
 

GaitSequenceDataset : retrieves the reuired joints data from the collected dataset.

preprocess : helps to reshape the data into valid sequences with given sequence length and splits data into training and validation

Convolutional_AE : It contains 1-dimensional convolutional autoencoder which has three convolutional, pooling and transpose convolutional layers. It is used to reconstruct the given data.
		  
train_conv_AE : It trains the model and gives us the training loss. It converts data into single dimension vector for the model.

loss function : MSE loss, it is similar for all the given encoding dimensions

optimizers: Adam, SGD, RMSprop