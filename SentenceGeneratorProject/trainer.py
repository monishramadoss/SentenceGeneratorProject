import torch
import progressbar
import torch.nn as nn
import torch.optim as ophim
import torch.nn.init as init
import torch.nn.functional as F
from nltk.tag import StanfordNERTagger
from torchtext import vocab
import nltk

nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')

import main
import sys
import os
import numpy as np
import json
from multiprocessing import Pool, Queue
import urllib.request
import csv
from model import SentenceGenerator

parser = main.parser
GeneratorDevice = main.GeneratorDevice
DiscriminatorDevice = main.DiscriminatorDevice

Res = 224
num_workers = 2
dtype = torch.float32
args, unknown = parser.parse_known_args()
DataFolder = args.input
batchSize = args.batchsize
model = args.model
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

def preProcessTask(Data, title, Max_val):
    TData = Data['data'][title]
    title  = TData['title']
    paragraphs = TData['paragraphs']
    Output = list()
    imposobru = False
    for context_qas in range(len(paragraphs)):
        Tags = torch.zeros(ContextLength, 36)
        Context = torch.zeros(ContextLength, 300)
        ctx = paragraphs[context_qas]['context']
        ctx_sents = ctx.split(r'.')
        context =  nltk.word_tokenize(ctx)
        for x in range(len(context)):
            Context[x] = get_glove_vec(context[x])                
        t = nltk.pos_tag(context)
        tagged = [tags[x[1]] if x[1] in tags.keys() else 36 for x in nltk.pos_tag(context)  ]        
        
        for x in range(len(tagged)):
            tmp = tagged[x]
            if(tmp != 36 ):
                Tags[x][tmp] = 1

        nerResult = nltk.ne_chunk(nltk.pos_tag(context))
        bioTagged = list()
        prevTag = 0      
            
        Context = torch.cat((Context,Tags), dim=1)
        
        if(Max_val < len(context)):
            Max_val = len(context)

        qas = paragraphs[context_qas]['qas']
        for qa in range(len(qas)):
            Question = torch.zeros(QuestionLength,300)
            Tags = torch.zeros(QuestionLength, 36)

            question = nltk.word_tokenize(qas[qa]['question'])
            tagged = [tags[x[1]] if x[1] in tags.keys() else 36 for x in nltk.pos_tag(question)  ]        

            for x in range(len(question)):
                try:
                    Question[x] = get_glove_vec(question[x])                
                except:
                    break

            for x in range(len(tagged)):
                try:
                    tmp = tagged[x]
                    if(tmp != 36 ):
                        Tags[x][tmp] = 1
                except:
                    pass

            Question = torch.cat((Question, Tags), dim=1)

            answers = qas[qa]['answers']
            imposobru = qas[qa]['is_impossible']
            sentence = ""
            Sentence = torch.zeros(SentenceLength, 300)
            Tags = torch.zeros(SentenceLength, 36)
            if(not imposobru):
                for ans in range(len(answers)):
                    Answer = torch.zeros(AnswerLength, 300)
                    answer_start = answers[ans]['answer_start']
                    answer_text = answers[ans]['text'].split(r' ')
                    
                    for sent in ctx_sents:
                        if(' '.join(answer_text) in sent):
                            sentence = nltk.word_tokenize(sent)
                            for x in range(len(sentence)):
                                try:
                                    Sentence[x]= get_glove_vec(sentence[x])
                                except:
                                    break

                            tagged = [tags[x[1]] if x[1] in tags.keys() else 36 for x in nltk.pos_tag(sentence)]    
                            for x in range(len(tagged)):
                                try:
                                    tmp = tagged[x]
                                    if(tmp != 36):
                                        Tags[x][tmp] = 1                                                                 
                                except:
                                    break
                            Sentence = torch.cat((Sentence, Tags), dim=1)
                            break 

                    for x in range(len(answer_text)):
                        try:
                            Answer[x] = get_glove_vec(answer_text[x])                      
                        except:
                            break
                    if(Sentence.shape[1] == 336 ):
                        Output.append((Context, Sentence, Question, Answer))    

    return (Output, imposobru)

def preprocess():
    
    TrainData = json.load(open(SquadTrainJson))          
    TestData = json.load(open(SquadTestJson))
    td = TrainData['data'][0]['paragraphs'][0]['qas']
    
    Context_Stack = list()
    Sentence_Stack = list()
    Question_Stack = list()
    Answer_Stack = list()
    
    for title in range(len(TrainData['data'])):
        tmp, Flag = preProcessTask(TrainData, title, Max_val)
        if(not Flag):
            for x in range(len(tmp)):            
                Context_Stack.append(tmp[x][0])
                Sentence_Stack.append(tmp[x][1])
                Question_Stack.append(tmp[x][2])
                Answer_Stack.append(tmp[x][3])                
            lens = len(Context_Stack)

    Context_Stack = torch.stack(Context_Stack)         
    Sentence_Stack = torch.stack(Sentence_Stack)
    Question_Stack = torch.stack(Question_Stack)
    Answer_Stack = torch.stack(Answer_Stack)
    
    Context_Stack = Context_Stack.numpy()
    Sentence_Stack = Sentence_Stack.numpy()
    Question_Stack = Question_Stack.numpy()
    Answer_Stack = Answer_Stack.numpy()

    np.save("./Dataset/Context.npy", Context_Stack)
    np.save("./Dataset/Sentence.npy", Sentence_Stack)
    np.save("./Dataset/Question.npy", Question_Stack)
    np.save("./Dataset/Answer.npy", Answer_Stack)

    

class SquadDataVecDataset(torch.utils.data.Dataset):
    def __init__(self):
        if(not os.path.exists('./Dataset/Context.npy') or not os.path.exists('./Dataset/Sentence.npy') or not os.path.exists('./Dataset/Question.npy') or not os.path.exists('./Dataset/Answer.npy')):
            preprocess()

        self.sentence = torch.from_numpy(np.load('./Dataset/Sentence.npy'))
        self.question = torch.from_numpy(np.load('./Dataset/Question.npy'))
        self.answer = torch.from_numpy(np.load('./Dataset/Answer.npy'))
        self.context = torch.from_numpy(np.load('./Dataset/Context.npy'))

    def __getitem__(self, idx):
        self.length = self.question.shape[0]
        return self.context, self.sentence, self.question, self.answer

    def __len__(self):
        return self.length


def train():
    dataset = SquadDataVecDataset()
    TrainDataLoader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    content_criterion = nn.MSELoss()
    model = SentenceGenerator()

    for epoch in range(10):
        for i, data in enumerate(TrainDataLoader):
            input = data



    return
    

def test():
    pass

