import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout = 0, num_layers = 1):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size, num_layers = self.num_layers, dropout = self.dropout, batch_first=True)

    def forward(self, x_input):
        lstm_out, self.hidden = self.lstm(x_input)
                
        return lstm_out, [self.hidden]

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
        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(-1), encoder_hidden_states[0])
        output = self.output_layer(lstm_out)
        
        return output, [self.hidden]

class seq2seq(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(seq2seq, self).__init__()

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

    def forward(self, inputs, targets, target_len, teacher_forcing_ratio = 0.5):
        #print(inputs.shape)
        batch_size = inputs.shape[0]
        input_size = inputs.shape[2]

        outputs = torch.zeros(batch_size, target_len, input_size)

        _, hidden = self.encoder(inputs)
        decoder_input = inputs[:,-1, :]
        
        for t in range(target_len): 
            out, hidden = self.decoder(decoder_input, hidden)
            out =  out.squeeze(1)
            if random.random() < teacher_forcing_ratio:
                decoder_input = targets[:, t, :]
            else:
                decoder_input = out
            outputs[:,t,:] = out

        return outputs

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