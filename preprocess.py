import numpy as np
import pandas as pd
import re
import nltk
re.compile('<title>(.*)</title>')
import string

#Creating the 'Sentiment Column'
#labeling scores:
#1, 2 -> Negative
#3 -> Neutral
#4, 5 -> Positive
def SetSentiment(row):
    if row['Score'] == 3:
        ans = 'Neutral'
    elif row['Score'] == 1 or row['Score'] == 2:
        ans = 'Negative'
    elif row['Score'] == 4 or row['Score'] == 5:
        ans = 'Positive'
    else:
        ans=-1
    return ans

def cleanreview(text):
    text = str(text).lower()
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


def stem_it(text):
    from nltk.stem.snowball import SnowballStemmer
    porter = SnowballStemmer("english")
    return [porter.stem(word) for word in text]

def stop_it(t):
    dt = [word for word in t if len(word)>2]
    return dt