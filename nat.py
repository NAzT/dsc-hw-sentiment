# -*- coding: utf-8 -*-

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

with open("38807051.txt") as f:
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

print(data[0])
print(len(data))
