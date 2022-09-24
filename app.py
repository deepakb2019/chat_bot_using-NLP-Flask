import os
import random
import json
import pickle
import numpy as np
import nltk

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from chat import get_response

from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
intents_booking = json.loads(open('intents_for_booking.json').read())
words = pickle.load(open('words.pkl', 'rb'))
labels = pickle.load(open('labels.pkl', 'rb'))
model = load_model('chatbot_model.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bagofwords(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bagofwords(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.1
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': labels[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(message):
    intents_list = predict_class(message)
    prob = float(intents_list[0]['probability'])
    tag = intents_list[0]['intent']
    # print(tag,end="0000")
    print(prob)
    list_of_intents = intents['intents']
    if prob > 0.8:
        for i in list_of_intents:
            # print(i['tag'])
            if i['tag'] == tag:
                return random.choice(i['responses'])
            # print(result)
    return "I do not understand..."


app = Flask(__name__)
CORS(app)


@app.get('/')
def index_get():
    return render_template('base.html')


@app.post('/predict')
def predict():
    text = request.get_json().get("message")
    # TODO: check if the text is valid
    text = text.lower()
    response = get_response(text)
    ints = predict_class(text)
    prob = float(ints[0]['probability'])
    intent = ints[0]['intent']
    print(prob)
    print(intent)
    if intent == 'booking' and prob > 0.8:
        message = {'answer': ['plz select the department <br> gynaecology <br>'"orthopaedics"" <br> neurology"]}
        return jsonify(message)
    elif intent == 'department_name' and prob > 0.8:
        message = {'answer': ['select the docter Name<br> DR.Ramesh <br> DR.Kiran  <br> DR.Jhon']}
        return jsonify(message)
    elif intent == 'docter_name' and prob > 0.8:
        message = {'answer': ['select the available time slots<br> 9:00am <br> 11:00am  <br> 2:00pm']}
        return jsonify(message)

    else:
        message = {'answer': response}
        return jsonify(message)


if __name__ == "__main__":
    app.run(host=os.getenv('IP', '0.0.0.0'),
            port=int(os.getenv('PORT', 4444)))
