import pandas as pd
import numpy as np
import spacy
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib
import string
from spacy.lang.en.stop_words import STOP_WORDS
from flask import Flask, request, jsonify, render_template
import nltk

# Load trained Pipeline
vectorization = joblib.load('vectorization.pickle')
model = joblib.load('fake_news_model.pkl')

stopwords = list(STOP_WORDS)


app = Flask(__name__) 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])



# Define the route for the prediction page
def predict():
    text = [str(x) for x in request.form.values()]
    text_vectors = vectorization.transform(text)
    text_vectors = text_vectors.toarray()


    # Reshape the input array
    text_vectors = np.array(text_vectors)

    predictions = model.predict(text_vectors)
    if predictions==0:
        return render_template('index.html', prediction_text='Fake News')
    else:
        return render_template('index.html', prediction_text='Not A Fake News')

if __name__ == '__main__':
    app.run(debug=True)
