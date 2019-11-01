# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 23:27:42 2019

@author: SurfaceBook2
"""
from pythainlp.tokenize import Tokenizer

pos= []
neg= []

with open("pos.txt", 'r') as f:
    for line in f:
        pos.append(line.rstrip())

with open("neg.txt", 'r') as f:
    for line in f:
        neg.append(line.rstrip())

url = '35213250'   
opinions = []
with open(url+".txt", 'r') as f:
    for line in f:
        opinions.append(line.rstrip())

mydict = pos+neg

tokenizer = Tokenizer(custom_dict=mydict, engine='newmm')

for opinion in opinions:
    neg_count = 0
    pos_count = 0
    print(opinion)
    text = tokenizer.word_tokenize(opinion)
    for word in text:
        if word in pos:
            pos_count = pos_count + 1
        if word in neg:
            neg_count = neg_count + 1

    if pos_count > neg_count:
        print('Positive')
    elif neg_count > pos_count:
        print('Negative')
    else:
        print('Neutral')