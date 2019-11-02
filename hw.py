# -*- coding: utf-8 -*-

from itertools import chain
import re
import pandas as pd
from pathlib import Path
import string

from pythainlp.tokenize import word_tokenize
from pythainlp import word_tokenize
from pythainlp.corpus import wordnet
from pythainlp.corpus import thai_stopwords

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
    tokens = word_tokenize(text, engine='newmm')

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

# print(texts[0])
# test_df = pd.DataFrame({ "category": "neu", "texts": texts })
# del texts # del labels
# test_df.to_csv("_test.csv", index=False)
# test_df.shape

data = []
clean_text = [clean_msg(str(txt)) for txt in texts]
for line in clean_text:
    data.append(split_word(str(line)))
tokens_list = data

print(tokens_list)
# print(data[1])
# print(len(data))

neg = []
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


ds = []
lbs = []

for i in training_data:
    ds.append(i[0])
    lbs.append(i[1])

print(training_data)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# tvec = TfidfVectorizer(analyzer=lambda x: x.split(','), )

tvec = TfidfVectorizer(tokenizer=word_tokenize)
# tokens_list_j = [','.join(tkn) for tkn in tokens_list]
X = tvec.fit_transform(ds)
y = lbs
print(X.shape)
print(tvec.get_feature_names())

print('X.shape', X.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestClassifier

text_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
text_classifier.fit(X_train, y_train)

input = "ผิดหวังด้วยคนค่ะ จะเดินออกจากเลนดีๆไม่ได้เลย ต้องเดินถอยหลังคอยโบกมือบ๊ายบาย ไหนจะตะโกนตามออกมาอีก ตั้งใจมาจับกี่ใบไม่เคยตามนั้นเลย เปลืองบัตรเพิ่มตลอด "
o = tvec.transform([input])
ooo = text_classifier.predict(o)

# print(o)
print(ooo)
