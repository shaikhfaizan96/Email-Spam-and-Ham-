import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


model = pickle.load(open(r"./model.pkl","rb"))
# bow = CountVectorizer(stop_words='english')
vectorizer= pickle.load(open(r"./vectorizer.pkl","rb"))

st.image("innomatics.jpeg")
st.header("EMAIL SPAM OR HAM")
#st.title("Email Spam/Ham Classifier")

# Input email text
Email = st.text_input("Paste the email here:")

# Check if the email input is not empty
if Email:
    # Transform the input email text to feature array
    data = vectorizer.transform([Email]).toarray()
    

    # Predict if the email is spam or ham
    spam_ham = model.predict(data)[0]

    # Display the prediction when the button is pressed
    if st.button('Submit'):
        if spam_ham == "spam":
            st.write("The email is:",  spam_ham)
            st.image("spam.jpeg",width=200)
        else:
            st.write("The email is:",  spam_ham)
            st.image("not spam.png",width=200)

