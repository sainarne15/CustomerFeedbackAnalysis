import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

import os
import pickle
import tensorflow as tf
import numpy as np
import random
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
from keras import models
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from sklearn.pipeline import Pipeline
import re
import pandas as pd
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import spacy
import streamlit as st



st.title('Customer Feedback Analysis')

# Load classification model
with st.spinner('Loading classification model...'):
    tokenizer = pickle.load(open('tokenizer.pickle', 'rb'))
    json_file = open('model_h8_text.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("my_h8_model_text.h5")

st.subheader('Please enter a feedback to classify:')

text = st.text_input('Feedback:')
if text != '':
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw,maxlen=200)

    # Make predictions
    with st.spinner('Predicting...'):
        prediction = int(loaded_model.predict(tw).round().item())

    # Show predictions
    if prediction == 1:
        output= 'This is a bad Product'
    else:
        output = 'This is a good Product'

    st.write('Prediction:')
    st.write(output)
    
    
    
