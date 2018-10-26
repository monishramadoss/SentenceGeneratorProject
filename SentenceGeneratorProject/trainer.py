import torch
import progressbar
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from nltk.tag import StanfordNERTagger
from torchtext import vocab
import nltk


import main
import sys
import os
import numpy as np
import json
import multiprocessing
from multiprocessing import Pool
from multiprocessing import Queue as PQueue
from queue import Queue
import urllib.request
import csv
import model

parser = main.parser
GeneratorDevice = main.GeneratorDevice
DiscriminatorDevice = main.DiscriminatorDevice

Res = 224
num_workers = 2
dtype = torch.float32
args, unknown = parser.parse_known_args()
DataFolder = args.input
batchSize = args.batchsize
model_type = args.model
Scale = args.Scale
DeviceConfig = args.deviceConfig
gan = args.gan
tileData = args.tileData
validateData = args.validateData
testData = args.testData
dtype = torch.float32
Max_val = 0
glove = vocab.GloVe(name='840B', dim=300)




SquadTrainJson = './Dataset/train-v2.0.json'
SquadTestJson = './Dataset/dev-v2.0.json'
StanfordParser = ['https://nlp.stanford.edu/software/stanford-ner-2018-02-27.zip',
                  'https://nlp.stanford.edu/software/stanford-postagger-2018-02-27.zip',
                  'https://nlp.stanford.edu/software/stanford-parser-full-2018-02-27.zip']

tags = {'CC':0, 'CD':1, 'DT':2, 'EX':3, 'FW':4, 'IN':5, 'JJ':6, 'JJR':7, 'JJS':8, 'LS':9, 'MD':10, 'NN':11, 'NNS':12, 'NNP':13, 'NNPS':14, 'PDT':15, 'POS':16, 'PRP':17, 'PRP$':18, 'RB':19, 'RBR': 20, 'RBS':21, 'RP':22, 'SYM':23, 'TO':24, 'UH':25, 'VB':26, 'VBD':27, 'VBG':28, 'VBN':29, 'VBP':30, 'VBZ':31, 'WDT':32, 'WP':33, 'WP$':34, 'WRB':35}

SentenceLength = 48
ContextLength = 800
QuestionLength = 32
AnswerLength = 16

def get_glove_vec(word):
    try:
        return glove.vectors[glove.stoi[word]]
    except:
        return torch.zeros(300)


def get_word_from_vec(vec, n=10):
    all_dists = [(w,torch.dist(vec, get_glove_vec(w))) for w in glove.itos]
    return sorted(all_dist, key=lambda t: t[1])[:n]

def preProcessTask(obj):
    context, sentence, question, answer = obj

    Context = torch.zeros(ContextLength, 300)
    Sentence = torch.zeros(SentenceLength, 300)
    Question = torch.zeros(QuestionLength, 300)
    Answer = torch.zeros(AnswerLength, 300)
    Tags_Context = torch.zeros(ContextLength, 36)
    Tags_Sentence = torch.zeros(SentenceLength, 36)
    Tags_Question = torch.zeros(QuestionLength, 36)
    Tags_Answer = torch.zeros(AnswerLength, 36)



    context =  nltk.word_tokenize(context)
    for x in range(len(context)):
        if(x < ContextLength):
            Context[x] = get_glove_vec(context[x])
    tagged = list()
    for x in nltk.pos_tag(context):
        if(x[1] in tags.keys() and len(tagged) < ContextLength):
            tagged.append(tags[x[1]])
        else:
            tagged.append(36)
    for x in range(len(tagged)):
        tmp = tagged[x]
        if(tmp != 36 ):
            Tags_Context[x][tmp] = 1
    Context = torch.cat((Context, Tags_Context), dim=1)



    sentence = nltk.word_tokenize(sentence)
    for x in range(len(sentence)):
        if(x < SentenceLength):
            Sentence[x] = get_glove_vec(sentence[x])
    tagged = list()
    for x in nltk.pos_tag(sentence):
        if(x[1] in tags.keys() and len(tagged) < SentenceLength):
            tagged.append(tags[x[1]])
        else:
            tagged.append(36)
    for x in range(len(tagged)):
        tmp = tagged[x]
        if(tmp != 36 ):
            Tags_Sentence[x][tmp] = 1
    Sentence = torch.cat((Sentence, Tags_Sentence), dim=1)



    question = nltk.word_tokenize(question)
    for x in range(len(question)):
        if(x < QuestionLength):
            Question[x] = get_glove_vec(question[x])
    tagged = list()
    for x in nltk.pos_tag(question):
        if(x[1] in tags.keys() and len(tagged) < QuestionLength):
            tagged.append(tags[x[1]])
        else:
            tagged.append(36)
    for x in range(len(tagged)):
        tmp = tagged[x]
        if(tmp != 36 ):
            Tags_Question[x][tmp] = 1
    Question = torch.cat((Question, Tags_Question), dim=1)
    

    
    answer = nltk.word_tokenize(answer)
    for x in range(len(answer)):
        if(x < AnswerLength):
            Answer[x] = get_glove_vec(answer[x])
    tagged = list()
    for x in nltk.pos_tag(answer):
        if(x[1] in tags.keys() and len(tagged) < AnswerLength):
            tagged.append(tags[x[1]])
        else:
            tagged.append(36)
    for x in range(len(tagged)):
        tmp = tagged[x]
        if(tmp != 36 ):
            Tags_Answer[x][tmp] = 1
    Answer = torch.cat((Answer, Tags_Answer), dim=1)


    return Context, Sentence, Question, Answer


class SquadDataVecDataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        self.Data = json.load(open(filename))       
        self.Output = list() #PQueue()
        

        if(not os.path.exists('./Dataset/SquadParsed.csv')):
            for x in range(len(self.Data['data'])):
                self.load_data(self.Data['data'][x])
        
            with open('./Dataset/SquadParsed.csv', 'w', newline='', encoding="utf-8") as csvfile:
                fieldnames = ['Context', 'Sentence', 'Question', 'Answer']
                writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for x in range(len(self.Output)):
                    data = self.Output[x]
                    writer.writerow(data)

        else:
            with open('./Dataset/SquadParsed.csv','r', newline='', encoding="utf-8") as csvfile:
                reader = csv.reader(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for row in reader:
                    self.Output.append(row)

    def __getitem__(self, idx):
        return preProcessTask(self.Output[idx])
       

    def __len__(self):
        return len(self.Data['data'])

    def load_data(self, data):          
        paragraphs = data['paragraphs']
        for context_qas in range(len(paragraphs)):
            context = paragraphs[context_qas]['context']
            qas = paragraphs[context_qas]['qas']
            ctx_sents = paragraphs[context_qas]['context'].split(r'.')
            for qa in range(len(qas)):
                answers = qas[qa]['answers']
                question = qas[qa]['question']
                imposobru = qas[qa]['is_impossible']
                if(not imposobru):
                    for ans in range(len(answers)):
                        answer_start = answers[ans]['answer_start']
                        answer_text = answers[ans]['text']
                        sentence = ''
                        for sent in ctx_sents:
                            if(answer_text in sent):
                                sentence = sent
                                break
                        if(sentence != ''):
                            tmp = (context, sentence, question, answer_text)
                            self.Output.append(tmp)
      
    
def Trainer(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=32):
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    SOS_token = 0

    input_length = input_tensor.shape[0]
    target_length = target_tensor.shape[0]

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size)
    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]])
    decoder_hidden = encoder_hidden

    use_target = True if random.random() < 0.5 else False

    if(use_target):
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]

    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            loss += criterion(decoder_output, target_tensor[di])

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()/target_length

def train():
    dataset = SquadDataVecDataset(SquadTestJson)
    TrainerDataLoader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle=True)
    paragraph_encoder = model.Encoder(336, 512)
    sentence_encoder = model.Encoder(336, 512)    
    decoder = model.Decoder(512, QuestionLength)

    criterion = nn.NLLLoss()
    
    paragraph_encoder_optimizer = optim.SGD(paragraph_encoder.parameters(), lr=0.01)
    sentence_encoder_optimizer = optim.SGD(sentence_encoder.parameters(), lr=0.01)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.01)

    for epoch in range(10):
        for i, data in enumerate(TrainerDataLoader):
            context = data[0]
            sentence = data[1]
            target = data[2]

            loss = Trainer(sentence, target, sentence_encoder, decoder, sentence_encoder_optimizer, decoder_optimizer, criterion )



    return
    

def test():
    dataset = SquadDataVecDataset(SquadTestJson)
    TrainDataLoader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    content_criterion = nn.MSELoss()
    model = SentenceGenerator()

    for epoch in range(10):
        for i, data in enumerate(TrainDataLoader):
            context = data[0]
            sentence = data[1]
            criterion = nn.NLLLoss()
            output = model(context, sentence)


    return

