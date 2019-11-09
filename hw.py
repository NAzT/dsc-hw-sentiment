# -*- coding: utf-8 -*-

from deepcut import DeepcutTokenizer
import deepcut
from itertools import chain
import re
import pandas as pd
from pathlib import Path
import string
from pprint import pprint

# from pythainlp.tokenize import word_tokenize
from pythainlp import word_tokenize
from pythainlp.corpus import wordnet
from pythainlp.corpus import thai_stopwords

tokenizer = DeepcutTokenizer(ngram_range=(1, 1),
                             max_df=1.0, min_df=0.0)
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import words

from stop_words import get_stop_words
import pandas as pd

nltk.download('words')
th_stop = tuple(thai_stopwords())
en_stop = tuple(get_stop_words('en'))
p_stemmer = PorterStemmer()

import re
import string


def clean_msg(msg):
    # ลบ text ที่อยู่ในวงเล็บ <> ทั้งหมด
    msg = re.sub(r'<.*?>', '', msg)

    # ลบ hashtag
    msg = re.sub(r'#', '', msg)
    msg = msg.replace('ชิมช็อปใช้เฟส', '')

    # ลบ เครื่องหมายคำพูด (punctuation)
    for c in string.punctuation:
        msg = re.sub(r'\{}'.format(c), '', msg)

    # ลบ separator เช่น \n \t
    msg = ' '.join(msg.split())

    return msg


def split_word(text):
    # tokens = word_tokenize(text, engine='deepcut')
    tokens = deepcut.tokenize(text)

    # Remove stop words ภาษาไทย และภาษาอังกฤษ
    tokens = [i for i in tokens if not i in th_stop and not i in en_stop]

    # หารากศัพท์ภาษาไทย และภาษาอังกฤษ
    # English
    tokens = [p_stemmer.stem(i) for i in tokens]

    # Thai
    tokens_temp = []
    for i in tokens:
        w_syn = wordnet.synsets(i)
        if (len(w_syn) > 0) and (len(w_syn[0].lemma_names('tha')) > 0):
            tokens_temp.append(w_syn[0].lemma_names('tha')[0])
        else:
            tokens_temp.append(i)

    tokens = tokens_temp

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

data = []
clean_text = [clean_msg(str(txt)) for txt in texts]
for line in clean_text:
    data.append(split_word(str(line)))
tokens_list = data

print(tokens_list)

negs = []
pos = []
neu = []
with open("neg_all.txt") as f:
    negs = [line.strip() for line in f.readlines()]

with open("pos_all.txt") as f:
    pos = [line.strip() for line in f.readlines()]

with open("neutral.txt") as f:
    neu = [line.strip() for line in f.readlines()]

# print(negs)
# print(pos)
import numpy

pos1 = ['pos'] * len(pos)
neg1 = ['neg'] * len(negs)
neu1 = ['neu'] * len(neu)

training_data = list(zip(pos, pos1)) + list(zip(negs, neg1)) + list(zip(neu, neu1))
# for
# print(training_data.shape)
print(neu)

ds = []
lbs = []

for i in training_data:
    ds.append(i[0])
    lbs.append(i[1])

# print(training_data)

from sklearn.feature_extraction.text import TfidfVectorizer

# tvec = TfidfVectorizer(analyzer=lambda x: x.split(','), )

tvec = TfidfVectorizer(tokenizer=deepcut.tokenize)
# tokens_list_j = [','.join(tkn) for tkn in tokens_list]
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

# pipeline = Pipeline([
#     ('vect', CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#     ('clf', SGDClassifier(tol=1e-3)),
# ])
#
# parameters = {
#     'vect__max_df': (0.5, 0.75, 1.0),
#     # 'vect__max_features': (None, 5000, 10000, 50000),
#     'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
#     # 'tfidf__use_idf': (True, False),
#     # 'tfidf__norm': ('l1', 'l2'),
#     'clf__max_iter': (20,),
#     'clf__alpha': (0.00001, 0.000001),
#     'clf__penalty': ('l2', 'elasticnet'),
#     # 'clf__max_iter': (10, 50, 80),
# }
#
# grid_search = GridSearchCV(pipeline, parameters, cv=5,
#                            n_jobs=-1, verbose=1)
#
# print("Performing grid search...")
# print("pipeline:", [name for name, _ in pipeline.steps])
# print("parameters:")
# pprint(parameters)
# t0 = time()
# grid_search.fit(X, y)
# print("done in %0.3fs" % (time() - t0))
# print()
#
# print("Best score: %0.3f" % grid_search.best_score_)
# print("Best parameters set:")
# best_parameters = grid_search.best_estimator_.get_params()

print('X.shape', X.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestClassifier

text_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
text_classifier.fit(X_train, y_train)

from sklearn.svm import SVC

text_classifier2 = SVC(gamma=10, C=3)
text_classifier2.fit(X_train, y_train)

input = "ความชั่วช้าซ้ำซ้อน"
o = tvec.transform([input])
ooo = text_classifier.predict(o)

score = text_classifier.score(X_test, y_test)
print('accuracy : ', score)

print(ooo)

with open("38807051.txt", mode='r', encoding='utf-8-sig') as f:
    for line in f:
        t = line.strip()
        print(t)
        o = tvec.transform([t])
        p1 = text_classifier.predict(o)
        p2 = text_classifier2.predict(o)
        print(p1, p2)
        print("=========================")

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier

from sklearn.base import BaseEstimator, TransformerMixin


class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.field]


class NumberSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.field]]


#
# classifier = Pipeline([
#     ('clf', [RandomForestClassifier()]),
# ])
#
# classifier.fit(X_train, y_train)
# preds = classifier.predict(X_test)
#
# print(preds)


# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

clfs = []
clfs.append(LogisticRegression())
clfs.append(SVC())
clfs.append(SVC())
clfs.append(KNeighborsClassifier(n_neighbors=3))
clfs.append(DecisionTreeClassifier())
clfs.append(RandomForestClassifier())
clfs.append(RandomForestClassifier())
clfs.append(GradientBoostingClassifier())
clfs.append(GradientBoostingClassifier())
pipeline = Pipeline([
    ('clf', LogisticRegression())  # step2 - classifier
])
pipeline.steps
from sklearn.model_selection import cross_validate

scores = cross_validate(pipeline, X_train, y_train)
print(scores)
for classifier in clfs:
    pipeline.set_params(clf=classifier)
    scores = cross_validate(pipeline, X_train, y_train)
    print('---------------------------------')
    print(str(classifier))
    print('-----------------------------------')
    for key, values in scores.items():
        print(key, ' mean ', values.mean())
        print(key, ' std ', values.std())
