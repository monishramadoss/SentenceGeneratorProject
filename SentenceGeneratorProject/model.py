import torch
import progressbar
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torchtext import vocab


class SentenceGeneration(nn.Module):
    def __init__(self, hidden_state=1, num_layers=2, ):
       
        super(SentenceGeneration, self).__init__()
        self.bi_Sentence_rnn_encoder = torch.nn.GRU(input_size=600, hidden_state=hidden_state, num_layers=num_layers, batch_first=False, bidirectional=True)
        self.bi_Paragraph_rnn_encoder = torch.nn.GRU(input_size=1024, hidden_state=hidden_state, num_layers=num_layers, batch_first=False, bidirectional=True)
        self.rnn_decoder = torch.nn.GRU(input_size=600, hidden_state=hidden_state, num_layer=num_layers, batch_first=False, bidirectional=False)
        self.attention = torch.nn.Softmax()
        self.linear = torch.nn.linear(600,600)

    def forward(self, context, sentence):
        
        Context_featureMap = self.bi_Paragraph_rnn_encoder(context)
        Sentence_featureMap = self.bi_Sentence_rnn_encoder(sentence)
        attention = self.attention(self.linear(Sentence_featureMap))

        torch.exp()
        Question_Decoding = self.rnn_decoder(Context_featureMap)
        return Question_Decoding


