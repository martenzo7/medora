import tensorflow as tf
import numpy as np
import pandas as pd
import json
import nltk
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder
import string
import random
import pickle

# Load the data
with open('content.json') as content:
    data1 = json.load(content)

# Getting data lists
tags = []
inputs = []
responses = {}
for intent in data1['intents']:
    responses[intent['tag']] = intent['responses']
    for line in intent['input']:
        inputs.append(line)
        tags.append(intent['tag'])

# Converting to DataFrame
data = pd.DataFrame({"inputs": inputs, "tags": tags})

# Pre-processing
data['inputs'] = data['inputs'].apply(lambda wrd: [ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['inputs'] = data['inputs'].apply(lambda wrd: ''.join(wrd))

# Tokenize the data
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['inputs'])
train = tokenizer.texts_to_sequences(data['inputs'])

# Apply padding
x_train = pad_sequences(train)

# Outputs encoding
le = LabelEncoder()
y_train = le.fit_transform(data['tags'])

# Creating model
input_shape = x_train.shape[1]
vocabulary = len(tokenizer.word_index) + 1
output_length = len(le.classes_)

input_layer = Input(shape=(input_shape,))
x = Embedding(vocabulary, 10)(input_layer)
x = LSTM(10, return_sequences=True)(x)
x = Flatten()(x)
x = Dense(output_length, activation='softmax')(x)
model = Model(input_layer, x)

# Compiling the model
model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

# Training the model
model.fit(x_train, y_train, epochs=200)

# Save the model
model.save('model/chatbot_model.h5')

# Save the tokenizer
with open('model/tokenizer.json', 'w') as f:
    json.dump(tokenizer.to_json(), f)

# Save the label encoder
with open('model/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("Model, tokenizer, and label encoder saved successfully.")

