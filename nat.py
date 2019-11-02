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
with open("neg.txt") as f:
    negs = [line.strip() for line in f.readlines()]

with open("pos.txt") as f:
    pos = [line.strip() for line in f.readlines()]

# print(negs)
# print(pos)
import numpy

pos1 = ['pos'] * len(pos)
neg1 = ['neg'] * len(negs)

training_data = list(zip(pos, pos1)) + list(zip(negs, neg1))
# for
# print(training_data.shape)


ds = []
lbs = []

for i in training_data:
    ds.append(i[0])
    lbs.append(i[1])

print(training_data)
# print(training_data)
# print('before training')
# vocabulary = set(chain(*[word_tokenize(i[0]) for i in training_data]))
# print('before training')
# feature_set = [({i: (i in split_word(sentence)) for i in vocabulary}, tag) for sentence, tag in
#                training_data]
# print('before training')
# print("feature_set", feature_set)
# from nltk import NaiveBayesClassifier as nbc
#
# print('start training')
# classifier = nbc.train(feature_set)
# print("done set")
# while True:
#     test_sentence = input('\nข้อความ : ')
#     featurized_test_sentence = {i: (i in word_tokenize(test_sentence.lower())) for i in vocabulary}
#     print("test_sent:", test_sentence)
#     print("tag:", classifier.classify(featurized_test_sentence))  # ใช้โมเดลที่ train ประมวลผล

from sklearn.feature_extraction.text import TfidfVectorizer

# from pythainlp.corpus import thai_stopwords
# th_stop = tuple(thai_stopwords())

# from pythainlp.ulmfit import process_thai
from sklearn.feature_extraction.text import TfidfVectorizer

# tvec = TfidfVectorizer(analyzer=lambda x: x.split(','), )

tvec = TfidfVectorizer(tokenizer=word_tokenize)
# tokens_list_j = [','.join(tkn) for tkn in tokens_list]
X = tvec.fit_transform(ds)
y = lbs
print(X.shape)
print(tvec.get_feature_names())
# y = numpy.asarray(all_labels)

# print('len1', len(all_data))
# print('len2', len(all_labels))

print('X.shape', X.shape)
# print('y.shape', y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestClassifier

text_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
text_classifier.fit(X_train, y_train)

# predictions = text_classifier.predict(X_test)
# print(predictions)
# print(X_test)
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#
# print(confusion_matrix(y_test, predictions))
# print(classification_report(y_test, predictions))
# print(accuracy_score(y_test, predictions))
# print(X_test)
input = "ผิดหวังด้วยคนค่ะ จะเดินออกจากเลนดีๆไม่ได้เลย ต้องเดินถอยหลังคอยโบกมือบ๊ายบาย ไหนจะตะโกนตามออกมาอีก ตั้งใจมาจับกี่ใบไม่เคยตามนั้นเลย เปลืองบัตรเพิ่มตลอด "
o = tvec.transform([input])
print(o)
ooo = text_classifier.predict(tvec.transform([input]))
print(ooo)

# tvec.transform(pos)
# tvec.transform(negs)

# tokens_list_j = [''.join(tkn) for tkn in tokens_list]
# t_feat = (tvec.fit_transform(clean_text))
# X = tvec.fit_transform(clean_text).toarray()
#
# print(t_feat.shape)
# print(tvec.vocabulary_)
# print(tvec.get_feature_names())

# print(X)

# print(tvec.get_feature_names())
# print(t_feat[:, :5].todense())
#
# from sklearn.cluster import KMeans

# n_clusters = 4
# #
# # kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
# # pred_y = kmeans.fit_predict(t_feat)
# # print(t_feat)
# # tvec.fit()
# tfidfconverter = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7, stop_words=th_stop)
# X = tfidfconverter.fit_transform(texts).toarray()
