#!/usr/bin/env python3

import joblib 
import pandas as pd
import nltk
import re
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

import twint
import nest_asyncio
from datetime import datetime
from datetime import timedelta

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize 


def load_model_vocab(model_name, vocab_name):
    model = joblib.load(model_name)
    vocab = joblib.load(vocab_name)['vector']

    return model, vocab

def get_live_tweets():
    nest_asyncio.apply()
    live = twint.Config()
    live.Search = "lockdown"
    live.Pandas = True
    live.Limit = 5
    live.Pandas_clean = True
    live.Geo = "28.5934,77.2223,2000km"
    live.Since = str(datetime.today() - timedelta(hours = 0.5))[:19]        # Do not change
    live.Until = str(datetime.now())[:19]                                   # Do not change
    live.Lang = 'en'
    live.Hide_output = True

    twint.run.Search(live)

    df_temp = pd.DataFrame(twint.storage.panda.Tweets_df)
    columns = ['date','tweet']
    df_temp = df_temp[columns]

    return df_temp


def cleansing(df):
    stemmer = SnowballStemmer("english", ignore_stopwords = True)
    text = []
    stop_words = set(stopwords.words('english'))
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    stemmer = nltk.stem.SnowballStemmer('english')

    for i in range(0, df.shape[0]):
        sentence = df['tweet'][i]
        sentence = tokenizer.tokenize(sentence)
        for x in sentence:
            x = stemmer.stem(x)
        sentence = [w for w in sentence if not w in stop_words]
        sentence = " ".join(sentence)
        sentence = sentence.lower()
        text.append(sentence)
    
    return text


def predictor(model, sentence, vocab = False):
    assert vocab is not False
    sentence = [sentence]
    sentence = vocab.transform(sentence).toarray()
    return model.predict(sentence)    



def live_analysis1(df, model, vocab):                   # For Data Preprocessing Methods by Nisarg    
    text = cleansing(df)
    res = lambda x: 'Positive' if (predictor(model, x, vocab) == 2) else ('Negative' if (predictor(model, x, vocab) == 0) else 'Neutral') 
    for i in range(0, df.shape[0]):
        print(i+1,'\n')
        print(df['tweet'][i],'\n')
        print('Sentiment:\t', res(text[i]))
        print('\n\n')    


def normalizer(tweet):
    ps = PorterStemmer() 
    wnl = WordNetLemmatizer()
    lemma = WordNetLemmatizer() 
    tweets = " ".join(filter(lambda x: x[0]!= '@', tweet.split()))
    tweets = re.sub(r'([^a-zA-Z0-9])', ' ', tweets)
    tweets=re.sub('[\s][a-z]{1,3}[\s]',' ',tweets)
    tweets=re.sub('^[a-z]{1,3}[\s]',' ',tweets)
    tweets=re.sub(r'[0-9_]','',tweets)
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '',tweets)
    tweets = tweets.lower()

    tweets = tweets.split()
    tweets = [word for word in tweets if not word in set(stopwords.words('english'))]
    tweets = [lemma.lemmatize(word) for word in tweets]
    
    tweets = " ".join(tweets)
        
    tweets= word_tokenize(tweets)   
    t=[]   
    for j in tweets:
           t.append(ps.stem(j))
           t.append(" ")
    tweets=" ".join(t)
    tweets=tweets.replace('ing','')
    tweets=tweets.replace('pic','')
    return tweets       


def live_analysis2(df, model, vocab):
    text = []
    for i in range(0, df.shape[0]):
        text.append(vocab.transform(pd.Series(normalizer(df['tweet'][i]))).toarray())
        
    res = lambda x: 'Positive' if (model.predict(x)[0] == 1) else ('Negative' if (model.predict(x)[0] == -1) else 'Neutral')
    
    for i in range(0, df.shape[0]):
        print("Tweet", i+1, "\n")
        print(df['tweet'][i], '\n')
        print('Sentiment:\t', res(text[i]))
        print("\n\n")     