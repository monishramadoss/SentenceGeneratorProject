import torch
import progressbar
import torch.nn as nn
import torch.optim as ophim
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
Output = Queue()


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


class SquadDataVecDataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        self.Data = json.load(open(filename))       
        for x in range(len(self.Data['data'])):
            self.load_data(self.Data['data'][x])
       
        with open('./Dataset/SquadParsed.csv', 'w', newline='') as csvfile:
            fieldnames = ['Context', 'Sentence', 'Question', 'Answer']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            while not Output.empty():
                data = Output.get()
                writer.writerow({'Context': data[0], 'Sentence': data[1], 'Question': data[2], 'Answer': data[3]})
                                
    def __getitem__(self, idx):

        context = list()
        sentence = list()
        question = list()
        answer = list()

        tmp, Flag = preProcessTask(self.Data, idx, Max_val)
        if(Flag):
            tmp, Flag = preProcessTask(self.Data, idx+1, Max_val)
            if(Flag):
                tmp, Flag = preProcessTask(self.Data, idx-1, Max_val)

        for c, s, q, a in tmp:
            context.append(c)
            sentence.append(s)
            question.append(q)
            answer.append(a)
        c = torch.cat(c, dim=0)
        s = torch.cat(c, dim=0)
        q = torch.cat(c, dim=0)
        a = torch.cat(c, dim=0)

    def __len__(self):
        return len(self.Data['data'])
    
    def threadTaskLoadData(self, obj):
        context, qas =- obj
        for qa in range(len(qas)):
            answers = qas[qa]['answers']
            question = qas[qa]['question']
            imposobru = qas[qa]['is_impossible']
            if(not imposobru):
                for ans in range(len(answers)):
                    answer_start = answers[ans]['answer_start']
                    answer_text =  answers[ans]['text']
                    sentence = ''
                    for sent in ctx_sents:
                        if(' '.join(answer_text) in sent):
                            sentence = sent
                            break
                    if(sentence != ''):
                        Output.put([context, sentence, question, answer_text])

    def load_data(self, data):          
        paragraphs = data['paragraphs']
        with Pool(processes=6) as pool:
            pool.map_async(self.threadTaskLoadData, [[paragraphs[context_qas]['context'], paragraphs[context_qas]['qas']] for context_qas in range(len(paragraphs))], 2)
            pool.close()
            pool.join()
            pool.terminate()

def train():
    dataset = SquadDataVecDataset(SquadTrainJson)
    TrainDataLoader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    content_criterion = nn.MSELoss()
    model = SentenceGenerator()

    #for epoch in range(1):
        #for i, data in enumerate(TrainDataLoader):
        #    input = data



    return
    

def test():
    dataset = SquadDataVecDataset(SquadTestJson)
    TrainDataLoader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    content_criterion = nn.MSELoss()
    model = SentenceGenerator()

    for epoch in range(10):
        for i, data in enumerate(TrainDataLoader):
            input = data



    return

