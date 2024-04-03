import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout = 0, num_layers = 1):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size, num_layers = self.num_layers, dropout = self.dropout, batch_first=True)

    def forward(self, x_input):
        #print("Encoder forward input shape: ", x_input.shape)
        lstm_out, self.hidden = self.lstm(x_input)
        
        #print("Encoder forward done!")
        return lstm_out, self.hidden

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout = 0, num_layers = 1):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size, num_layers = self.num_layers, dropout = self.dropout, batch_first=True)
        self.output_layer = nn.Linear(self.hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_input, encoder_hidden_states):
        lstm_out, self.hidden = self.lstm(x_input, encoder_hidden_states)
        output = self.output_layer(lstm_out)
        
        return output, self.hidden

class seq2seq(nn.Module):
    def __init__(self, input_window, output_window, input_size, hidden_size, num_layers, dropout):
        super(seq2seq, self).__init__()

        self.input_window = input_window
        self.output_window = output_window
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.encoder = Encoder(input_size = self.input_size, 
                                    hidden_size = self.hidden_size, 
                                    num_layers = self.num_layers, 
                                    dropout = self.dropout)
        self.decoder = Decoder(input_size = self.input_size, 
                                    hidden_size = self.hidden_size, 
                                    num_layers = self.num_layers, 
                                    dropout = self.dropout)

    def forward(self, inputs, targets, target_len, teacher_forcing_ratio = 1.0):
        batch_size = inputs.shape[0]

        outputs = torch.zeros(batch_size, target_len, 1)

        # 0~8 timestep for encoder, 9~14 : timestep for decoder        
        encoder_input = inputs[:,0:self.input_window,:]
        decoder_input = inputs[:,(-1)*self.output_window:,:]

        _, hidden = self.encoder(encoder_input)

        # Full teacher forcing for decoder or not.
        # Teacher Forcning(Give the answer)
        if random.random() <= teacher_forcing_ratio:
            print("Teacher Forcing!")
            out, hidden = self.decoder(decoder_input, hidden)

            outputs[:, :, :] = out[:, :, :]
        
        # Non-Teacher Forcing(Using past output)
        else:
            print("Non-Teacher forcing")
            
            for t in range(target_len): #0 1 2 3 4 5
                out, hidden = self.decoder(decoder_input, hidden)

                if t < target_len - 1 : 
                    decoder_input[:, t + 1, 0] = out[:, t , 0]

                outputs[:, t, 0] = out[:, t, 0]

        return outputs.squeeze()


        
        '''
        # Each input teacher forcing for decoder.
        for t in range(target_len):
            print("Seq2Seq decoder part start")
            
            out, hidden = self.decoder(decoder_input, hidden)
            print("Decoder output shape : ", out.shape)
            out =  out.squeeze(1)
            if random.random() <= teacher_forcing_ratio:
                print("Teacher Forcing : target shape", targets.shape)
                print(decoder_input[0:5, t, 0])
                decoder_input = targets[:, t, :]
            else:
                print("Not Teacher Forcing : target shape", targets.shape)
                print(decoder_input[0:5, t, 0])
                print(targets[:, t])
                decoder_input = out
                

            outputs[:,t,:] = out
        '''

    def predict(self, inputs, target_len):
        inputs = inputs.unsqueeze(0)
        self.eval()
        batch_size = inputs.shape[0]
        input_size = inputs.shape[2]
        outputs = torch.zeros(batch_size, target_len, input_size)
        _, hiddens = self.encoder(inputs)
        decoder_input = inputs[:,-1, :]
        for t in range(target_len): 
            out, hidden = self.decoder(decoder_input, hiddens)
            out =  out.squeeze(1)
            decoder_input = out
            outputs[:,t,:] = out
        return outputs.detach().numpy()[0,:,0]