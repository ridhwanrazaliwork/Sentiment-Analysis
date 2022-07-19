#%%
# Tokenizer
import pickle
import re
import json
import os
import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#%%
# Model Loading
# Tokenizer
TOKENIZER_PATH = os.path.join(os.getcwd(),'model', 'tokenizer.json')
MODEL_PATH = os.path.join(os.getcwd(),'model','model.h5')
OHE_PATH = os.path.join(os.getcwd(),'model', 'ohe.pkl')

with open(TOKENIZER_PATH,'r') as file:
    loaded_tokenizer = json.load(file)

tokenizer = tokenizer_from_json(loaded_tokenizer)

# Model

model = load_model(MODEL_PATH)

# model.summary()

# Ohe
with open(OHE_PATH,'rb') as file:
    ohe = pickle.load(file)
# Model
# %%

# new_review = ["Watched It in IMAX, wonderful, epic! no SPOILERS for anyone here. The Movie is truly not as good as Deadpool Part 1, as Part 1 had a trustworthy storyline,  A hero, a Villian, hero has to take revenge from the Villian who did him wrong, But Deadpool 2 is world's Apart, There is no specific Villian Segregated in the Movie, watch it and you'll come to know, The action scenes are totally awesome, while some of the scenes including Gore and blood are cut, The movie is one of the Most ENTERTAINING movie ever produced. It isn't the best superhero movie, it can't even get close to Infinity War. But it is the most enjoyable movie i've ever seen. Lots Of Comedy, Lots of Action, lots of fourth wall breaking humor, Lots of twists and cameos, It's unpredictable and truly enjoyable, but The only drawback of watching this movie is that your lungs will start hurting, beware not to Die Laughing! and also that all entertaining things end within the blink of an eye. Yes it seems like a 10 minute film to me, compared to Infinity war, as It is enjoyed and lacks even a bit of seriousness, whereas Infinity war still has an impression of a Lifelong Infinite Movie That Never Ends. At the End If You're Wondering whether you should watch it or not just Go For It, It is the most entertaining movie in The Recent Times, The Cameos and twists in The Story will blow your mind!"]
# new_review = [input('Please type your review here')]
# new_review = st.text_input('Please type your review here')

# for index, text in enumerate(new_review):
#     # to remove html tags
#     new_review = re.sub('<.*?>','',text)
#     new_review = re.sub('[^a-zA-Z]',' ',new_review).lower().split()

# # DAta preprocessing
# new_review = tokenizer.texts_to_sequences(new_review)
# new_review = np.reshape(new_review,(1,len(new_review)))
# new_review = pad_sequences(new_review,maxlen=178,padding='post', truncating='post')
# # %% model prediction

# outcome = model.predict(new_review)

# print(f'This review is {ohe.inverse_transform(outcome)[0][0]}')
# %%
# Streamlit
"""
# Sentiment Analysis Tools

"""
st.header('Background')

st.markdown('''Sentiment analysis, often known as opinion mining, is a natural language processing (NLP) method for identifying the positivity, negativity, or neutrality of data. Businesses frequently do sentiment analysis on textual data to track the perception of their brands and products in customer reviews and to better understand their target market.''')
with st.form("Sentiment Analysis app"):

    # Every form must have a submit button.
    new_review = st.text_input('Please type your review here')
    for index, text in enumerate(new_review):
            # to remove html tags
            new_review = re.sub('<.*?>','',text)
            new_review = re.sub('[^a-zA-Z]',' ',new_review).lower().split()
            # Data preprocessing
    new_review = tokenizer.texts_to_sequences(new_review)
    new_review = np.reshape(new_review,(1,len(new_review)))
    new_review = pad_sequences(new_review,maxlen=178,padding='post', truncating='post')
    outcome = model.predict(new_review)
    review_outcome = ohe.inverse_transform(outcome)[0][0]
    submitted = st.form_submit_button("Submit")
    if submitted:
        if review_outcome == 'positive':
            st.write('This review is positive')
            st.balloons()
        if review_outcome == 'negative':
            st.write('This review is negative')
            st.snow()