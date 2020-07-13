from cloudant import Cloudant #!!COMMENTED FROM BHUSHAN'S CODE!!
import time
import atexit #!!COMMENTED FROM BHUSHAN'S CODE!!
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.stem import WordNetLemmatizer #word stemmer class
lemma = WordNetLemmatizer()
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pickle
import joblib
import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import stopwords
# import re
import string
import os


"# -- coding: utf-8 --"
# vector = pickle.load(open('vec', 'rb'))
# vectorizer=vector['vector']
vectorizer = pickle.load(open("vector.pickel", "rb"))
w = pickle.load(open('train', 'rb'))
x_train=w['x_train']
vectorizer.fit(x_train)
print(vectorizer)
print(len(vectorizer.get_feature_names()))
filename = 'logR.sav'

# filename='logR.sav'
loaded_model = joblib.load(open(filename, 'rb'))
def calctime(a):
    return time.time() - a


positive = 0
negative = 0
neutral = 0
compound=0

count = 0
initime = time.time()

ps = PorterStemmer()
wnl = WordNetLemmatizer()


ckey = 'WnyAgUaacX1YheRSJqwMhhZgR'
csecret = 'LzHg7GuAfJNIsHRpRXEk72TaEjcG5RL9yl85c0rbI1V1pg6rHQ'
atoken = "1125091796046843905-DNeIxEe9RNwlwzZZXwXEW3VJFlv7Az"
asecret = "n3Yc9GzA2Saa6LNPZ5465WdQNj06G6hBrqcWnpwkc4jCb"

js=pd.DataFrame(columns=['text','labels'])
tl=[]
class listener(StreamListener):

    def on_data(self, data):
        global initime
        global tl
        all_data = json.loads(data)
        tweet = all_data["text"]

        # username=all_data["user"]["screen_name"]
        # tweet = " ".join(re.findall("[a-zA-Z]+", tweet))
        #

        # print(tweet)
        global ps
        global wnl

        sequencePattern = r"(.)\1\1+"
        seqReplacePattern = r"\1\1"
        tweets = " ".join(filter(lambda x: x[0] != '@', tweet.split()))
        tweets = re.sub(r'([^a-zA-Z0-9])', ' ', tweets)
        tweets = re.sub(r'[^\w\s]', '', tweets)
        #     tweets=re.sub('[\s][a-z]{1,3}[\s]',' ',tweets)
        #     tweets=re.sub('^[a-z]{1,3}[\s]',' ',tweets)
        tweets = re.sub(r'[0-9_]', '', tweets)
        tweets = re.sub(r'^https?:\/\/.*[\r\n]*', '', tweets)
        tweets = re.sub(r"http\S+", "", tweets)
        tweets = tweets.lower()
        tweets = re.sub(sequencePattern, seqReplacePattern, tweets)
        #     print(tweets)
        tweets = tweets.split()
        tweets = [word for word in tweets if not word in set(stopwords.words('english'))]
        tweets = [lemma.lemmatize(word) for word in tweets]

        tweets = " ".join(tweets)

        #     print(tweets[0])
        #     tweets=tweets.split(" ")

        tweets = word_tokenize(tweets)
        #     print(tweets)
        t = []
        for j in tweets:
            #
            t.append(ps.stem(j))
            #            t.append(wnl.lemmatize(j))
            t.append(" ")

        tweets = " ".join(t)
        #     tweets = tweets.split()
        tweets = tweets.replace('ing', '')
        tweets = tweets.replace('pic', '')
        tweets = tweets.replace('com', '')

        # l = vectorizer.transform(tweets).toarray()
        # tweets='fuck corona'
        k = pd.Series(tweets)
        print(k)

        l = vectorizer.transform(k).toarray()
        # print(l)
        m=loaded_model.predict(l)
        print(m[0])
        t = int(calctime(initime))
        # blob = TextBlob(tweet.strip())
        print(t)
        # print(loaded_model)
        # print(vectorizer)
        tl.append({'tweet':tweet,'label':m[0]})
        global positive
        global negative
        global neutral
        global count
        global compound
        count = count + 1
        # senti = 0
        # for sen in blob.sentences:
        #     senti = senti + sen.sentiment.polarity
        #     if sen.sentiment.polarity >= 0:
        #         positive = positive + sen.sentiment.polarity
        #     else:
        #         negative = negative + sen.sentiment.polarity
        # compound = compound + senti
        # print
        # count
        # print
        if m[0]==1:
            positive=positive+1
        elif m[0]==-1:
            negative=negative+1
        else:
            neutral=neutral+1

        print("pos ",positive)
        print("neg",negative)
        print("neu",neutral)
        k={"pos":positive,"neg":negative,"time":t,"details":tl}
        s=pd.DataFrame(k)
        try:
            s.to_json('count.json')
            tl=[]
        except:
            print("couldnt")
        # u=positive+negative+neutral
        sen=[positive,negative,neutral]
        # print(sen)
        xsen=['positive','negative','neutral','time']
        tweets.strip()
        # print
        # senti
        # print
        # t
        # print
        # str(positive) + ' ' + str(negative) + ' ' + str(neutral)
        # print(len(t))
        # plt.axis([ 0, 70,0,220])
        # plt.xlabel('Time')
        # plt.ylabel('Sentiment')
        # plt.plot([t],[positive],'go',[t] ,[negative],'ro',[t],[neutral],'bo')
        # plt.plot([t],[u])
        width = 0.35
        # plt.bar(xsen,sen,width = 0.35,color='r')

        # plt.show()
        # plt.pause(0.0001)
        # temp={'text':tweet,'labels':m}
        # js=pd.DataFrame(temp)
        #
        # js.append(temp,ignore_index=True)
        # try:
        #     print(js)
        #     js.to_json('obj.json')
        # except:
        #     print("cannot")
        #     pass
        if count == 10:
            return False

        else:
            return True

    def on_error(self, status):
        print(status)

while True:
    try:
        auth = OAuthHandler(ckey, csecret)
        auth.set_access_token(atoken, asecret)
        twitterStream = Stream(auth, listener(count),lang='en',geocode="22.3511148,78.6677428,1km")
        twitterStream.filter(track=["#IndiaFightsCorona","covid19 india","corona india","#covid19#india","corona warriors","#cluelessbjp"])
    except Exception as e:
        time.sleep(5)
