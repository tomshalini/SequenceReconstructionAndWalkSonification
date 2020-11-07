import sys
import torch
from IPython import display

def train(model, train_dataloader, optimizer, criterion, clip, device, epoch, num_epochs, summary, loss_display_interval):
    print('In train..')
    
    model.train()
    num_batches = len(train_dataloader)

    for batch_idx, data in enumerate(train_dataloader):
        optimizer.zero_grad()
        sequences = data['sequence'].permute(1, 0, 2).to(device)
        piano_rolls = data['piano_roll'].permute(2,0,1).to(device)
        sequence_lengths = data['sequence_length']
        sequence_lengths, sort_indices = sequence_lengths.sort(dim=0, descending=True)
        sequences = torch.index_select(sequences, 1, sort_indices.to(device))
        piano_rolls = piano_rolls.contiguous().view(-1).to(device)

        predicted_piano_roll = model(sequences, sequence_lengths)

        loss = criterion(predicted_piano_roll.to(device), piano_rolls.to(device))

        if(batch_idx % loss_display_interval == 0):
            display.clear_output(True)
            step = epoch * num_batches + batch_idx
            summary.add_scalar("Training Loss", loss.item(), step)
            print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(epoch,num_epochs, batch_idx, num_batches))
            print('Training Loss: {:.4f}'.format(loss))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        #break

def validate(model, validation_dataloader, criterion, device, epoch, num_epochs, summary, loss_display_interval):
    print('In validation..')

    model.eval()
    num_batches = len(validation_dataloader)

    total_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(validation_dataloader):

            sequences = data['sequence'].permute(1, 0, 2).to(device)
            piano_rolls = data['piano_roll'].permute(2,0,1).to(device)
            sequence_lengths = data['sequence_length'].to(device)

            sequence_lengths, sort_indices = sequence_lengths.sort(dim=0, descending=True)
            sequences = torch.index_select(sequences, 1, sort_indices.to(device))

            piano_rolls = piano_rolls.contiguous().view(-1)

            predicted_piano_roll = model(sequences, sequence_lengths)

            loss = criterion(predicted_piano_roll.to(device), piano_rolls.to(device))
            
            if(batch_idx % loss_display_interval == 0):
                display.clear_output(True)
                step = epoch * num_batches + batch_idx
                summary.add_scalar("Validation Loss", loss.item(), step)
                print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(epoch,num_epochs, batch_idx, num_batches))
                print('Validation Loss: {:.4f}'.format(loss))

            total_loss += loss
            #break

    return total_loss/num_batches