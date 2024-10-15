import json
import string
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import pickle

# Load the model
model = load_model('model/chatbot_model.h5')

# Load the tokenizer
with open('model/tokenizer.json') as f:
    tokenizer_json = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_json)

# Load the label encoder
with open('model/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Load the responses dictionary (for demonstration, you can copy from build.py)
with open('content.json') as content:
    data1 = json.load(content)

responses = {}
for intent in data1['intents']:
    responses[intent['tag']] = intent['responses']

input_shape = model.input_shape[1]  # Get the input shape of the model

while True:
    prediction_input = input('you: ')  # Get user input
    prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
    prediction_input = ''.join(prediction_input)

    # Tokenize and pad the input for prediction
    seq = tokenizer.texts_to_sequences([prediction_input])  # Tokenize the input
    padded_seq = pad_sequences(seq, maxlen=input_shape)  # Pad the sequence

    output = model.predict(padded_seq, verbose=0)  # Make prediction
    output = output.argmax()  # Get the index of the highest probability

    response_tag = le.inverse_transform([output])[0]  # Convert index back to tag
    print('bot: ', random.choice(responses[response_tag]))  # Select a random response
    if response_tag == "goodbye":
        break

