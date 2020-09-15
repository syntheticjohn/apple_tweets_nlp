"""
Topic modeling on user tweets to Apple Support --
Script for data preprocessing
Used to create aggregated df / pickles to be used for model creation
This script is designed to be run in a Jupyter notebook or the command line,
with mongodb pre-stored with the tweet dataset
"""

import numpy as np
import pandas as pd
import datetime as dt
import random
from collections import Counter
from scipy import sparse
import pickle
import os
import re
import string
from pymongo import MongoClient

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tokenize import MWETokenizer
import spacy
import gensim
import emoji
from spellchecker import SpellChecker
from textblob import TextBlob
from langdetect import detect

#====================================================================
### Load tweet data from mongodb (where the tweet data has been stored)
#====================================================================

# set up client instance
client = MongoClient()

# set up db instance
db = client.customersupport

# check collections in db
db.list_collection_names()

# load collection into dataframe 
cursor = db.tweets.find()
df = pd.DataFrame(list(cursor))
df.to_pickle('data/customer_tweets.pkl')

# drop column _id
df = df.drop(['_id'], axis=1)

# focus on tweets to and from apple support only
df = df[(df.author_id == 'AppleSupport') | (df.text.str.contains('@applesupport', na=False, flags=re.IGNORECASE, regex=True))]

# remove outbound messages that are not from apple support 
df = df[~((df.inbound == 'False') & (df.author_id != 'AppleSupport'))]

#====================================================================
### Cleaning the text
#====================================================================

# convert created at column to datetime type
df['created_at'] = pd.to_datetime(df['created_at'], format='%a %b %d %H:%M:%S +0000 %Y')

# add date only column
df['date_only'] = df['created_at'].dt.normalize()

# fix word lengthening, such as the word 'amazingggggg'
def reduce_lengthening(text):
    """
    converts words with more than 3 consecutive letters into correct spelling form
    """
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

df['text_clean'] = df.text.apply(lambda x: reduce_lengthening(x))

# lower case text
df.text_clean = df.text_clean.str.lower()

# remove punctuation
punc = (lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', str(x))) 
df.text_clean = df.text_clean.map(punc)

# remove curly open and closing quotes (for both single and double quotes)
# single curly open quote
df.text_clean = df.text_clean.map(lambda x: re.sub("‘", ' ', str(x)))
# single curly closing quote
df.text_clean = df.text_clean.map(lambda x: re.sub("’", ' ', str(x)))
# double curly open quote
df.text_clean = df.text_clean.map(lambda x: re.sub("“", ' ', str(x)))
# double curly closing quote
df.text_clean = df.text_clean.map(lambda x: re.sub("”", ' ', str(x)))

# remove numbers
num = (lambda x: re.sub('\w*\d\w*', ' ', str(x)))
df.text_clean = df.text_clean.map(num)

# convert slang / abbreviated phrases to words, such as brb to be right back    
chat_words_map_dict = {}
chat_words_list = []
with open('data/chat_words_str.txt', 'r') as file:
    chat_words_str = file.read()
for line in chat_words_str.split("\n"):
    if line != "":
        cw = line.split("=")[0]
        cw_expanded = line.split("=")[1]
        chat_words_list.append(cw)
        chat_words_map_dict[cw] = cw_expanded
chat_words_list = set(chat_words_list)

def chat_words_conversion(text):
    """
    converts common chat acronyms into it's fully spelled out phrase  
    """
    new_text = []
    for w in text.split():
        if w.upper() in chat_words_list:
            new_text.append(chat_words_map_dict[w.upper()])
        else:
            new_text.append(w)
    return " ".join(new_text)
df.text_clean = df.text_clean.apply(lambda x: chat_words_conversion(x))

# correct spelling using spell checker
spell = SpellChecker()
def correct_spellings(text):
    """ 
    converts incorrectly spelled words into correct spelling 
    """
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)        
df.text_clean = df.text_clean.apply(lambda x: correct_spellings(x))

# identify and retain non-english tweets only using text blob
def detect_lang(x):   
    """
    detects if text is of the english language
    """
    b = TextBlob(x)
    return b.detect_language()   
df['text_lang'] = df.text_clean.apply(lambda x: detect_lang(x))
df = df[df.text_lang == 'eng']

# remove stop words
stop = stopwords.words('english')
df.text_clean = df.text_clean.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# lemmatize
lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    """ 
    converts words into its lemmatized form 
    """
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])
df.text_clean = df.text_clean.apply(lambda text: lemmatize_words(text))

# remove emoji 
def give_emoji_free_text(text):
    """ 
    deletes emojis from text
    """
    allchars = [str for str in text] 
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)]) 
    return clean_text
df.text_clean = df.text_clean.apply(lambda x: give_emoji_free_text(x))

# remove urls
def remove_urls(text):
    """ 
    deletes url links from text
    """
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)
df.text_clean = df.text_clean.apply(lambda x: remove_urls(x))

# pickle dataframe for modeling
df.to_pickle('data/tweet_clean.pkl')

#====================================================================
### Create a dataframe for user tweets to Apple Support only
#====================================================================

# create dataframe for tweets sent to Apple Support only, keeping each tweet as an individual document
df_tweet_user = df.copy()
df_tweet_user = df_tweet_user[df_tweet_user.inbound == 'True']

# pickle dataframe for modeling
df_tweet_user.to_pickle('data/df_tweet_user.pkl')

#====================================================================
### Create a dataframe for a user's initial tweet to Apple Support
#====================================================================

# create dataframe for a user's initial tweet to Apple Support (i.e., excluding replies)
df_first_tweet_user = df.copy()
df_first_tweet_user = df_first_tweet_user[df_first_tweet_user.inbound == 'True']

# filter dataframe to first tweet per user only
df_first_tweet_user = df_first_tweet_user.loc[df_first_tweet_user.groupby('author_id').
                                              created_at.idxmin()].reset_index(drop=True)

# pickle dataframe for modeling
df_first_tweet_user.to_pickle('data/df_first_tweet_user.pkl')

#====================================================================
### Create a dataframe that combines user tweets from a conversation into one document
#====================================================================

# create new dataframe where all tweets from a user are combined (tweet conversation treated as a document)
df_convo_user = df.copy()
df_convo_user = df_convo_user[df_convo_user.inbound == 'True']

# aggregate text by user
df_orig_text = df_convo_user.groupby(['author_id'])['text'].apply(' '.join).reset_index()

# get min and max date per user
df_min_max_date = df_convo_user.groupby('author_id').agg({'date_only':['min', 'max']}).reset_index()

# add column names to min and max date
df_min_max_date.columns = ['_'.join(col).strip() for col in df_min_max_date.columns.values]

# aggregate text clean by user
df_convo_user = df_convo_user.groupby(['author_id'])['text_clean'].apply(' '.join).reset_index()

# merge df convo user with aggregated text
df_convo_user = pd.merge(df_convo_user, df_orig_text[['author_id', 'text']], how='left', on='author_id')

# merge df convo user with min and max date
df_convo_user = pd.merge(df_convo_user, df_min_max_date[['author_id_', 'date_only_min', 'date_only_max']], how='left', left_on='author_id', right_on='author_id_')
df_convo_user = df_convo_user.drop(['author_id_'], axis=1)
df_convo_user = df_convo_user[['author_id', 'text', 'text_clean', 'date_only_min', 'date_only_max']]

# pickle dataframe for modeling
df_convo_user.to_pickle('data/df_convo_user.pkl')