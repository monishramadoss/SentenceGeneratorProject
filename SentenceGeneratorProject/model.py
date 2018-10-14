import torch
import progressbar
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torchtext import vocab


class SentenceGeneration(nn.Module):
    def __init__(self, hidden_state=1, num_layer=1, ):
       
        super(SentenceGeneration, self).__init__()
        self.bi_rnn_encoder = torch.nn.GRU(input_size=300, hidden_state=1, num_layers=1, batch_first=False, bidirectional=True)
        self.rnn_decoder = torch.nn.GRU(input_size=300, hidden_state=1, num_layer=1, batch_first=False, bidirectional=False)


    def forward(self, x):
        pass

