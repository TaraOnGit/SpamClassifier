import streamlit as st
import pickle

import nltk
import string
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)

  y = []
  for i in text:
    if i.isalnum():
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    if i not in nltk.corpus.stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  text = y[:]
  y.clear()

  ps = PorterStemmer()
  for i in text:
    y.append(ps.stem(i))

  return ' '.join(y)

#Pickles
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


st.title('Spam Classifier')
message = st.text_area('Enter Your Message')

if st.button('Check for Spam'):
  #1. Preprocess
  transformed_msg = transform_text(message)
  #2. Vectorize
  vector_msg = tfidf.transform([transformed_msg])
  #3. Predict
  pred = model.predict(vector_msg)[0]
  #4. Display
  if pred == 1:
      st.header('Spam')
  else:
      st.header('Not Spam')