
# Import the needed packages

import time
import pymongo
import pandas as pd
import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pymongo import MongoClient
import sqlalchemy
from sqlalchemy import create_engine


# 1. EXTRACT the tweets from mongodb

# Create connections to databases
client = pymongo.MongoClient(host="mongodb", port=27017)
db = client.twitter

time.sleep(10) # seconds

#Extract tweets 
extracted_tweets = list(db.tweets.find())
print(extracted_tweets)

# 2. TRANSFORM the data

#Sentiment analysis
s = SentimentIntensityAnalyzer()
neg = []
neu = []
pos = []
compound = []
tweets_EM = []
for tweet in extracted_tweets:
    tweet_EM = tweet ['text']
    score = s.polarity_scores(f'{tweet_EM}')
    neg.append(score['neg'])
    neu.append(score['neu'])
    pos.append(score['pos'])
    compound.append(score['compound'])
    tweets_EM.append(tweet_EM)
    print(tweet_EM)
    print(score)

df_EM = pd.DataFrame({'tweet':tweets_EM, 'neg': neg, 'neutral':neu, 'pos':pos, 'compound': compound})
print(df_EM)
print(df_EM.info())

# 3. LOAD the data into postgres

# To onnect to Postgres
HOST = 'postgresdb'
PORT = '5432' #port inside the container
DATABASE = 'postgres'
USER = 'postgres'
PASSWORD = 'postgres'

# Load the tweets and the resulting sentimental analysis
conn_string = f'postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}'
try:
    engine = sqlalchemy.create_engine(conn_string, echo=False)
    engine.connect()
    df_EM.to_sql('em_tweets', engine, if_exists='replace')
    logging.critical('We get in!')
except:
    logging.critical('Postgres load failed!')

