# WalkingSonification - single LSTM Neural Network Autoencoder

Takes walking motion data which contains joints information like flexion angles between the hip and upper-leg, upper leg and lower-leg, and lower-leg and feet.
 

GaitSequenceDataset : retrieves the reuired joints data from the collected dataset.

preprocess : helps to reshape the data into valid sequences with given sequence length and splits data into training and validation

single_LSTM : It contains a single LSTM cell autoencoder. Its encoding dimensions are 32, 64 and128. It is used to reconstruct the given data.
		  
train_ae_single : It trains the model and gives us the training loss. It converts data into single dimension vector for the model.

loss function : RMSE loss, it is different for all the given optimizers but similar for different encoding dimensions.

optimizers: Adam, SGD, RMSprop