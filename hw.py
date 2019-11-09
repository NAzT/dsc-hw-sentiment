# -*- coding: utf-8 -*-

import deepcut
from itertools import chain
import re
import pandas as pd
from pathlib import Path
import string
from pprint import pprint

from pythainlp.tokenize import word_tokenize
# from pythainlp import word_tokenize
from pythainlp.corpus import wordnet
from pythainlp.corpus import thai_stopwords

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import words

from stop_words import get_stop_words
import pandas as pd

# from pythainlp.tokenize import dict_word_tokenize, create_custom_dict_trie


nltk.download('words')
th_stop = tuple(thai_stopwords())
print(th_stop)
en_stop = tuple(get_stop_words('en'))
p_stemmer = PorterStemmer()

import re
import string


def mytokenizer(text):
    return word_tokenize(text, engine='newmm')


def clean_msg(msg):
    # ลบ text ที่อยู่ในวงเล็บ <> ทั้งหมด
    msg = re.sub(r'<.*?>', '', msg)

    # ลบ hashtag
    msg = re.sub(r'#', '', msg)

    # ลบ เครื่องหมายคำพูด (punctuation)
    for c in string.punctuation:
        msg = re.sub(r'\{}'.format(c), '', msg)

    # ลบ separator เช่น \n \t
    msg = ' '.join(msg.split())

    return msg


def split_word(text):
    tokens = mytokenizer(text)
    #
    # Remove stop words ภาษาไทย และภาษาอังกฤษ
    # tokens = [i for i in tokens if not i in th_stop and not i in en_stop]
    # #
    # # หารากศัพท์ภาษาไทย และภาษาอังกฤษ
    # # English
    # tokens = [p_stemmer.stem(i) for i in tokens]
    #
    # # Thai
    # tokens_temp = []
    # for i in tokens:
    #     w_syn = wordnet.synsets(i)
    #     if (len(w_syn) > 0) and (len(w_syn[0].lemma_names('tha')) > 0):
    #         tokens_temp.append(w_syn[0].lemma_names('tha')[0])
    #     else:
    #         tokens_temp.append(i)
    #
    # tokens = tokens_temp

    # ลบตัวเลข
    tokens = [i for i in tokens if not i.isnumeric()]

    # ลบช่องว่าง
    tokens = [i for i in tokens if not ' ' in i]

    return tokens


texts = []
labels = []

with open("35213250.txt", mode='r', encoding='utf-8-sig') as f:
    for line in f:
        texts.append(line.strip())

# data = []
# clean_text = [clean_msg(str(txt)) for txt in texts]
# for line in clean_text:
#     data.append(split_word(str(line)))
# tokens_list = data
#
# print(tokens_list)

negs = []
pos = []
neu = []
with open("neg_all.txt") as f:
    negs = [line.strip() for line in f.readlines()]

with open("pos_all.txt") as f:
    pos = [line.strip() for line in f.readlines()]

with open("neutral.txt") as f:
    neu = [line.strip() for line in f.readlines()]

import numpy

pos1 = ['pos'] * len(pos)
neg1 = ['neg'] * len(negs)
neu1 = ['neu'] * len(neu)

training_data = list(zip(pos, pos1)) + list(zip(negs, neg1)) + list(zip(neu, neu1))

ds = []
lbs = []

for i in training_data:
    ds.append(i[0])
    o = ''
    # if i[1] == "pos":
    #     o = 0
    # if i[1] == "neg":
    #     o = 1
    # if i[1] == "neu":
    #     o = 2

    # lbs.append(o)
    lbs.append(i[1])

# print(training_data)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# tvec = TfidfVectorizer(analyzer=lambda x: x.split(','), )


# tokens_list_j = [','.join(tkn) for tkn in tokens_list]

tvec = CountVectorizer(tokenizer=split_word)
X = tvec.fit_transform(ds)
y = lbs
print(X.shape)
print(X.toarray())

# print(tvec.get_feature_names())
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from time import time

print('X.shape', X.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestClassifier

# text_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
# text_classifier.fit(X_train, y_train)
#
# from sklearn.svm import SVC
#
# text_classifier2 = SVC(gamma=10, C=3)
# text_classifier2.fit(X_train, y_train)
#
# input = "ความชั่วช้าซ้ำซ้อน"
# o = tvec.transform([input])
# ooo = text_classifier.predict(o)
#
# score = text_classifier.score(X_test, y_test)
# print('accuracy : ', score)

# print(ooo)

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier

from sklearn.base import BaseEstimator, TransformerMixin

# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

clfs = []
clfs.append(LogisticRegression())
clfs.append(DecisionTreeClassifier())
clfs.append(RandomForestClassifier())
clfs.append(MLPClassifier(alpha=1, max_iter=1000))
# clfs.append(AdaBoostClassifier())
# clfs.append(GradientBoostingClassifier())
# clfs.append(SVC())
# clfs.append(KNeighborsClassifier(n_neighbors=5))
pipeline = Pipeline([
    ('clf', LogisticRegression())  # step2 - classifier
])
pipeline.steps
from sklearn.model_selection import cross_validate

scores = cross_validate(pipeline, X_train, y_train)
print(scores)
models = []
for classifier in clfs:
    pipeline.set_params(clf=classifier)
    scores = cross_validate(pipeline, X_train, y_train)
    print('---------------------------------')
    print(str(classifier))
    print('-----------------------------------')
    for key, values in scores.items():
        print(key, ' mean ', values.mean())
        print(key, ' std ', values.std())
    models.append(classifier.fit(X_train, y_train))

# listword = ['แมว', "ดี"]
# data_dict = create_custom_dict_trie(listword)
# print(data_dict)
with open("39285983.txt", mode='r', encoding='utf-8-sig') as f:
    for line in f:
        t = line.strip()
        print(t)
        print(split_word(t))
        # print(deepcut.tokenize(t))
        o = tvec.transform([t])
        results = []
        for m in models:
            # print(m.predict(o))
            print(m.predict(o), str(m).split("(")[0])
        print("=========================")
