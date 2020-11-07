import sys
import torch
from IPython import display

def train(encoder, decoder, train_dataloader, encoder_optimizer, decoder_optimizer, criterion, clip, device, epoch, num_epochs, summary, loss_display_interval):
    print('In train..')
    
    encoder.train()
    decoder.train()
    num_batches = len(train_dataloader)

    for batch_idx, data in enumerate(train_dataloader):
        
        sequences = data['sequence'].permute(1, 0, 2).to(device)
        piano_roll = data['piano_roll'].permute(1, 0, 2).to(device)
        
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        x ,(hidden_state, cell_state)= encoder(sequences)
        prediction = decoder(hidden_state)

        loss = criterion(prediction, piano_roll)

        if(batch_idx % loss_display_interval == 0):
            display.clear_output(True)
            step = epoch * num_batches + batch_idx
            summary.add_scalar("Training Loss", loss.item(), step)
            print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(epoch,num_epochs, batch_idx, num_batches))
            print('Training Loss: {:.4f}'.format(loss))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)

        encoder_optimizer.step()
        decoder_optimizer.step()
        # break

def validate(encoder, decoder, validation_dataloader, criterion, device, epoch, num_epochs, summary, loss_display_interval):

    encoder.eval()
    decoder.eval()

    num_batches = len(validation_dataloader)
    total_loss = 0

    for batch_idx, data in enumerate(validation_dataloader):
        
        sequences = data['sequence'].permute(1, 0, 2).to(device)
        piano_roll = data['piano_roll'].permute(1, 0, 2).to(device)

        x ,(hidden_state, cell_state) = encoder(sequences)
        prediction = decoder(hidden_state)

        loss = criterion(prediction, piano_roll)
        total_loss += loss.item()


        if(batch_idx % loss_display_interval == 0):
            display.clear_output(True)
            step = epoch * num_batches + batch_idx
            summary.add_scalar("Validation Loss", loss.item(), step)
            print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(epoch,num_epochs, batch_idx, num_batches))
            print('Validation Loss: {:.4f}'.format(loss))
        
        # break
    return total_loss/num_batches