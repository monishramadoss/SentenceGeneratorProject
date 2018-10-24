import torch
import progressbar
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torchtext import vocab


class SentenceGenerator(nn.Module):
    def __init__(self, hidden_state=1, num_layers=2, ):
       
        super(SentenceGenerator, self).__init__()
        self.bi_Sentence_rnn_encoder = torch.nn.GRU(input_size=600, hidden_state=hidden_state, num_layers=num_layers, batch_first=False, bidirectional=True)
        self.bi_Paragraph_rnn_encoder = torch.nn.GRU(input_size=1024, hidden_state=hidden_state, num_layers=num_layers, batch_first=False, bidirectional=True)
        self.rnn_decoder = torch.nn.GRU(input_size=600, hidden_state=hidden_state, num_layer=num_layers, batch_first=False, bidirectional=False)
        self.attention = torch.nn.Softmax()
        self.linear = torch.nn.linear(600,600)

    def forward(self, context, sentence):
        
        Context_featureMap = self.bi_Paragraph_rnn_encoder(context)
        Sentence_featureMap = self.bi_Sentence_rnn_encoder(sentence)
        attention = self.attention(self.linear(Sentence_featureMap))

        Sentence_Context_fedatureMap = torch.cat((Context_featureMap, Sentence_featureMap), dim=0)
        
        Question_Decoding = self.rnn_decoder(Sentence_Context_fedatureMap)
        return Question_Decoding


