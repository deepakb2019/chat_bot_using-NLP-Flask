import json
import pickle
import random


import nltk
import numpy as np
# nltk.download()
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

# config = tensorflow.config()
# config.gpu_options.allow_growth = True
# session = tensorflow.Session(config=config)
lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
labels = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        # print(word_list)
        words.extend(word_list)
        # print(words)
        documents.append((word_list, intent['tag']))
        # print(documents)
        if intent['tag'] not in labels:
            labels.append(intent['tag'])
            # print(labels)
# print(training_data)
# print(len(words))
# print()

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]  # edited
# print(len(words))
words = sorted(list(set(words)))  # edited

labels = sorted(list(set(labels)))  # edited

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(labels, open('labels.pkl', 'wb'))
training_set = []
output_empty = [0] * len(labels)
# print(output_empty)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    # print(word_patterns)
    # print()
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    # print(word_patterns,end='*')
    for word in words:
        # print(word,end='#')
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[labels.index(doc[1])] = 1
    training_set.append([bag, output_row])
# print(word_patterns)
# print(len(words))
# print((bag))
# print(output_row)
random.shuffle(training_set)
training_set = np.array(training_set)

train_x = list(training_set[:, 0])
train_y = list(training_set[:, 1])
model = Sequential()
model.add(Dense(112, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(24, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
# model = Sequential()
# model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=500, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print("Done")
