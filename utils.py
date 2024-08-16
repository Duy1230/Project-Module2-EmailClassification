import pandas as pd
import numpy as np
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')

def lower_case(text):
    return text.lower()

def remove_punctuation(text):
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)

def remove_numbers(text):
    return text.translate(str.maketrans("", "", string.digits))

def tokenize(text):
    return nltk.word_tokenize(text)

def remove_stop_words(text):
    stop_words = set(nltk.corpus.stopwords.words("english"))
    return [word for word in text if word not in stop_words]

def stemming(text):
    stemmer = nltk.PorterStemmer()
    return [stemmer.stem(word) for word in text]

def process_text(text):
    text = lower_case(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = tokenize(text)
    text = remove_stop_words(text)
    text = stemming(text)
    return text

def create_dictionary(messages):
    dictionary = []
    for message in messages:
        for word in message:
            if word not in dictionary:
                dictionary.append(word)
    return dictionary

def create_features(message, dictionary):
    features = [0] * len(dictionary)
    for word in message:
        if word in dictionary:
            features[dictionary.index(word)] = 1
    return features