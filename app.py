import pickle
import streamlit as st
from utils import process_text, create_features

# Load model
MODEL_PATH = "models\\model.pkl"
DICTIONARY_PATH = "models\\dictionary.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(DICTIONARY_PATH, "rb") as f:
    dictionary = pickle.load(f)


# Header
st.header("Project-Module2: Email Classifier using Naïve Bayes Classifier 📧")

st.markdown("Please enter your email: ")

email = st.text_input("")

classifier = st.button("Classify")
if classifier:
    if not email:
        st.warning("Please enter your email")
    else:
        input = process_text(email)
        input = create_features(input, dictionary)
        prediction = model.predict([input])[0]

        print(prediction)
        if prediction == 'ham':
            prediction = "Not Spam"
        else:
            prediction = "Spam"

        st.success("Email classified as: " + prediction)
