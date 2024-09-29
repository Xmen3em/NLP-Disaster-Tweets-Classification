# Important imports
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from app import app
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import string

# Adding path to config
app.config['INITIAL_FILE_UPLOADS'] = 'app/static/uploads'
app.config['EXISTNG_FILE'] = 'app/static/original'
app.config['GENERATED_FILE'] = 'app/static/generated'

def remove_url(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)

def remove_html(text):
    html_tag = re.compile(r'<.*?>')
    return html_tag.sub(r'', text)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_punct(text):
    table = str.maketrans('','',string.punctuation)
    return text.translate(table)

import tensorflow_hub as hub
def get_tfhub_model(model_link, model_name, model_trainable = False):
    return hub.KerasLayer(model_link, 
                         trainable = model_trainable,
                         name = model_name, 
                         dtype = tf.string)

# get universal sentence encoder model
encoder_link = 'https://kaggle.com/models/google/universal-sentence-encoder/frameworks/TensorFlow2/variations/universal-sentence-encoder/versions/2'

encoder_name = 'universal_sentence_encoder'
encoder_trainable = False

encoder = get_tfhub_model(encoder_link, model_name = encoder_name, model_trainable = encoder_trainable)

def build_pretrained_model():
    # Define kernel initializer & input layer
    initializer = tf.keras.initializers.HeNormal(seed = 42)
    # Correct the input shape to be 1D (None,)
    tweets_input = layers.Input(shape=(), dtype=tf.string)
    
    # Generate Embeddings using a Lambda layer to call the encoder
    tweets_embedding = layers.Lambda(lambda x: encoder(x), output_shape=(512,))(tweets_input)
    
    # Feed Embeddings to a Bidirectional LSTM
    expand_layer = layers.Lambda(lambda embed: tf.expand_dims(embed, axis = 1))(tweets_embedding)
    bi_lstm = layers.Bidirectional(layers.LSTM(128, kernel_initializer = initializer), 
                                   name = 'bidirection_lstm')(expand_layer)
    
    # Feed LSTM output to classification head
    dropout_layer = layers.Dropout(0.25)(bi_lstm)
    dense_layer = layers.Dense(128, activation = 'relu', kernel_initializer = initializer)(dropout_layer)
    output_layer = layers.Dense(1, activation='sigmoid', 
                                kernel_initializer = initializer, 
                                name = 'output_layer')(dense_layer)
    
    return tf.keras.Model(inputs = [tweets_input], 
                          outputs = [output_layer], 
                          name = 'use_model')
model = build_pretrained_model()
model.load_weights('USE_Model.weights.h5')

def model_predict(text):

    text = remove_url(text)
    text = remove_html(text)
    text = remove_emoji(text)
    text = remove_punct(text)   
    
    preds = model.predict(tf.convert_to_tensor([text], dtype = tf.string))
    predictions = tf.round(preds).numpy().astype(int)
    print(predictions)    
    return predictions


@app.route('/', methods=['Get'])
def index():
    #home page
    return render_template('home.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        tweet = request.form['tweet']
        my_pred = model_predict(tweet)
        return render_template('result.html', prediction = my_pred)

@app.route('/home', methods=['Get'])
def home():
    #home page
    return render_template('home.html')


# Main function
if __name__ == '__main__':
    app.run(debug=True)
