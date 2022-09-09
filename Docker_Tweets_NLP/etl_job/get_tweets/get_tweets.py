#Import needed packages

# To collect tweets
import tweepy
import os
import logging

# MongoDB
import pymongo

# For authenticiation
BEARER_TOKEN = os.getenv('BEARER_TOKEN')
client = tweepy.Client(bearer_token=BEARER_TOKEN)

if client:
    logging.critical("\nAuthentication OK")
else:
    logging.critical('\nVerify your credentials')

# Search for recent tweets on - Emmanuel Macron 

# Defining a query search string, english, no retweets:
query = 'Emmanuel Macron lang:en -is:retweet'

# Connect to mongo DB 

client_mongo = pymongo.MongoClient(host="mongodb", port=27017)
db = client_mongo.twitter

#Collect tweets
EM_tweets = tweepy.Paginator(method = client.search_recent_tweets,tweet_fields=['text'], query=query).flatten(limit=30)
print('Paginator Emmanuel Macron tweets check')
for tweet in EM_tweets:
    print(tweet)
    db.tweets.insert_one(dict(tweet))
    print('Tweet saved to Mongo DB!\n')

