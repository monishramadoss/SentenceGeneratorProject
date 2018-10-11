import torch
import progressbar
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
import json
from torchtext import vocab

import main
import sys
import os
import numpy as np

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

glove = vocab.GloVe(name='840B', dim=300)

SquadTrainJson = './Dataset/train-v2.0.json'
SquadTestJson = './Dataset/dev-v2.0.json'

def get_glove_vec(word):
    return glove.vectors[glove.stoi[word]]

def get_word_from_vec(vec, n=10):
    all_dists = [(w,torch.dist(vec, get_glove_vec(w))) for w in glove.itos]
    return sorted(all_dist, key=lambda t: t[1])[:n]

def preProcessTask(Data, title):
    TData = Data['data'][title]
    title  = TData['title']
    paragraphs = TData['paragraphs']
    Output = list()
    
    for context_qas in range(len(paragraphs)):
        context = paragraphs[context_qas]['context']
        qas = paragraphs[context_qas]['qas']
        Context = [get_glove_vec(word) for word in context.split(r' ')]     
        Ctx = torch.zeros((1024, 300))
        Ctx.new_tensor(Context, device=torch.device('cuda:0'))
        for qa in range(len(qas)):
            question = qas[qa]['question']
            answers = qas[qa]['answers']
            for ans in range(len(answers)):
                Ans = torch.zeros((64,300))
                answer_start = answers[ans]['answer_start']
                answer_text = answers[ans]['text']
                Answer_Text = [get_glove_vec(word) for word in answer_text.split(r' ')]  
                Ans.new_tensor(Answer_Text, device=torch.device('cuda:0'))     
                Output.append(torch.cat((Ctx, Ans), dim=0))
    return 
                

def preprocess():
    
    TrainData = json.load(open(SquadTrainJson))
    TestData = json.load(open(SquadTestJson))

    td = TrainData['data'][0]['paragraphs'][0]['qas']

    for title in range(len(TrainData['data'])):
        preProcessTask(TrainData, title)
                    
    print()

def train():
    preprocess()
    

def test():
    pass

