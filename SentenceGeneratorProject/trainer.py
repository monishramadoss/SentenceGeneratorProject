import torch
import progressbar
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torchtext import vocab

import main
import sys
import os
import numpy as np
import json
from multiprocessing import Pool, Queue

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
    for context_qas in range(len(paragraphs)):
        context = paragraphs[context_qas]['context'].split(r' ')
        if(Max_val < len(context)):
            Max_val = len(context)
        qas = paragraphs[context_qas]['qas']
        Context = torch.zeros(600, 300)
        for x in range(len(context)):
            Context[x] = get_glove_vec(context[x])        
        for qa in range(len(qas)):
            question = qas[qa]['question']
            answers = qas[qa]['answers']
            for ans in range(len(answers)):
                Answer = torch.zeros(64,300)
                answer_start = answers[ans]['answer_start']
                answer_text = answers[ans]['text'].split(r' ')
                for x in range(len(answer_text)):
                    Answer[x] = get_glove_vec(answer_text[x])
                Output.append(torch.cat((Context, Answer), dim=0))
    return Output
                

def preprocess():
    
    TrainData = json.load(open(SquadTrainJson))
    TestData = json.load(open(SquadTestJson))

    td = TrainData['data'][0]['paragraphs'][0]['qas']
    p = Pool(1)    
    #args = [(TrainData, title, Max_val) for title in range(len(TrainData['data']))]      
    #Output = p.starmap(preProcessTask, args)
    #p.join()

    Output = list()
    for title in range(len(TrainData['data'])):
       Output.append(preProcessTask(TrainData, title, Max_val))
                    
    print()

def train():
    preprocess()
    

def test():
    pass

