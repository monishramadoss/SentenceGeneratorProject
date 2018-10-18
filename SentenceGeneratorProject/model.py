import torch
import progressbar
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torchtext import vocab


class SentenceGeneration(nn.Module):
    def __init__(self, hidden_state=1, num_layers=1, ):
       
        super(SentenceGeneration, self).__init__()
        self.bi_rnn_encoder = torch.nn.GRU(input_size=300, hidden_state=hidden_state, num_layers=num_layers, batch_first=False, bidirectional=True)
        self.rnn_decoder = torch.nn.GRU(input_size=300, hidden_state=hidden_state, num_layer=num_layers, batch_first=False, bidirectional=False)


    def forward(self, x):
        
        Context_featureMap = self.bi_rnn_encoder(x)
        Question_Decoding = self.rnn_decoder(Context_featureMap)

    return Question_Decoding


