import os
import warnings
import absl.logging

# Suppress TensorFlow and Abseil warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN if desired
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU usage
absl.logging.set_verbosity(absl.logging.ERROR)  # Suppresses Abseil logs
warnings.filterwarnings("ignore", category=UserWarning, module='absl')  # Suppress specific warnings

import json
import string
import random
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Load the model
model = load_model('model/chatbot_model.h5')

# Load the tokenizer
with open('model/tokenizer.json') as f:
    tokenizer_json = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_json)

# Load the label encoder
with open('model/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Load the responses dictionary
with open('content.json') as content:
    data1 = json.load(content)

# Create a dictionary to hold responses for each tag
responses = {}
for intent in data1['intents']:
    responses[intent['tag']] = intent['responses']

input_shape = model.input_shape[1]  # Get the input shape of the model

# Interactive loop for user input
while True:
    prediction_input = input('you: ')  # Get user input
    # Preprocess the input
    prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
    prediction_input = ''.join(prediction_input)

    # Tokenize and pad the input for prediction
    seq = tokenizer.texts_to_sequences([prediction_input])  # Tokenize the input
    padded_seq = pad_sequences(seq, maxlen=input_shape)  # Pad the sequence

    # Make prediction
    output = model.predict(padded_seq, verbose=0)  # Use verbose=0 to suppress prediction logs
    output = output.argmax()  # Get the index of the highest probability

    # Convert index back to tag and respond
    response_tag = le.inverse_transform([output])[0]
    print('bot: ', random.choice(responses[response_tag]))  # Select a random response
    if response_tag == "goodbye":  # Break the loop if the tag is "goodbye"
        break

