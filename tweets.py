import numpy as np
import pandas as pd
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

tweets = pd.read_csv \
    ("https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv")

print(tweets.describe())
print(tweets.iloc[:, 10])

print(tweets.iloc[:, 1])
