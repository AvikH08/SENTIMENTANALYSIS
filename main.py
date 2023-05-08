from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import keras
from keras.models import load_model
from preprocess import *
from nltk.tokenize import word_tokenize
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json
model = load_model('model.h5')
print('MODEL SUCCESSFULLY LOADED')

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)


# with open('tokenizer.json') as f:
#     tokenizer_json = f.read()
# tokenizer = tokenizer_from_json(tokenizer_json)
print('tokenizer successfully loaded')

app = Flask(__name__, static_url_path='/static', static_folder='static')

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Sentiment analysis page
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['inp']
    data = cleanreview(data)
    print(data)
    data = word_tokenize(data)
    print(data)
    data = stem_it(data)
    print(data)
    data = stop_it(data)
    print(data)
    data = ' '.join(data)
    print(data)
    X = tokenizer.texts_to_sequences([data])
    print(X)
    X = pad_sequences(X, padding = 'post', maxlen = 100)
    print(X)
    pred = model.predict(X)
    print(pred)
    index = np.argmax(pred[0])
    pospred = pred[0, 2]
    neutralpred = pred[0, 1]
    negpred = pred[0, 0]
    if index == 0:
        pred = 'Negative'
    elif index == 1:
        pred = 'Neutral'
    else:
        pred = 'Positive'
    return render_template('index.html', result = pred, pospred = pospred*100, negpred = negpred*100, neutralpred = neutralpred*100)


if __name__ == '__main__':
    app.run(debug=True)
