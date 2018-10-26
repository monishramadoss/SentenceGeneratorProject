import torch
import progressbar
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torchtext import vocab


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder,self).__init__() 
        self.hidden_size = hidden_size
        self.embedding  = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size)
       
    def forward(self, input, hidden):
        #embedding = self.embedding(input).view(1, 1, -1)
        output = input #embedding 
        output, hidden = self.rnn(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)




class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, drouput_p=0.3, max_len = 32):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attn = nn.Linear(hidden_size * 2, max_len)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(drouput_p)
        self.rnn = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_output):
       embedding = self.embedding(input).view(1, 1, -1)
       embedding = self.dropout(embedding)
       attn_weight = F.softmax(self.attn(torch.cat((embedding[0], hidden[0]), 1)), dim=1)
       attn_applied = torch.bmm(attn_weight.unsqueeze(0), encoder_output.unsqueeze(0))

       output = torch.cat((embedding[0], attn_applied[0]), 1)
       output = self.attn_combine(output).unsqueeze(0)

       output = F.relu(output)
       output, hidden = self.gru(output, hidden)

       output = F.log_softmax(self.out(output[0]) , dim=1)
       return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


