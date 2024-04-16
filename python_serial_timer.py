#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import json, nltk
import matplotlib.pyplot as plt
# from wordcloud import WordCloud
import seaborn as sns
# nltk.download('wordnet')   # for Lemmatization
from timeit import default_timer as timer

get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


#begin

start=timer()
# total_data = pd.read_csv("file:///c:/Users/fazal/Desktop/Uni/project/dataset/twitter-airline-sentiment/Tweets.csv", encoding="ISO-8859-1")
# total_data=total_data[['tweet_id','airline_sentiment', 'text']]
# total_data = total_data.rename(columns={'airline_sentiment': 'Sentiment','tweet_id':'ItemID','text':'SentimentText'})
total_data = pd.read_csv("file:///c:/Users/fazal/Desktop/Uni/project/dataset/trainingandtestdata/training.1600000.processed.noemoticon.csv", encoding="ISO-8859-1")
total_data=total_data[['Sentiment', 'SentimentText']]
# total_data=total_data.append(total_data, ignore_index=True)
# total_data=total_data.append(total_data, ignore_index=True)
# total_data=total_data.append(total_data, ignore_index=True)
# total_data=total_data.append(total_data, ignore_index=True)
with open('c:/Users/fazal/Desktop/Uni/project/dataset/contractions.json', 'r') as f:
    contractions_dict = json.load(f)
contractions = contractions_dict['contractions']
pd.set_option('display.max_colwidth', None)
# total_data.head()
# tweet = total_data.columns.values[2]
# sentiment = total_data.columns.values[1]
tweet = total_data.columns.values[1]
sentiment = total_data.columns.values[0]

total_data.info()
total_data.dropna()
def emoji(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :') , :O
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\)|:O)', ' positiveemoji ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' positiveemoji ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' positiveemoji ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-; , @-)
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;|@-\))', ' positiveemoji ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:, :-/ , :-|
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:|:-/|:-\|)', ' negetiveemoji ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' negetiveemoji ', tweet)
    return tweet

import re

def process_tweet(tweet):
    tweet = tweet.lower()                                             # Lowercases the string
    tweet = re.sub('@[^\s]+', '', tweet)                              # Removes usernames
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' ', tweet)   # Remove URLs
    tweet = re.sub(r"\d+", " ", str(tweet))                           # Removes all digits
    tweet = re.sub('&quot;'," ", tweet)                               # Remove (&quot;) 
    tweet = emoji(tweet)                                              # Replaces Emojis
    tweet = re.sub(r"\b[a-zA-Z]\b", "", str(tweet))                   # Removes all single characters
    for word in tweet.split():
        if word.lower() in contractions:
            tweet = tweet.replace(word, contractions[word.lower()])   # Replaces contractions
    tweet = re.sub(r"[^\w\s]", " ", str(tweet))                       # Removes all punctuations
    tweet = re.sub(r'(.)\1+', r'\1\1', tweet)                         # Convert more than 2 letter repetitions to 2 letter
    tweet = re.sub(r"\s+", " ", str(tweet))                           # Replaces double spaces with single space    
    return tweet

total_data['processed_tweet'] = np.vectorize(process_tweet)(total_data[tweet])
# total_data.head(10)

#count vectorizer
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer(ngram_range=(1,2))    # Unigram and Bigram
# count_vectorizer = CountVectorizer()
final_vectorized_data = count_vectorizer.fit_transform(total_data['processed_tweet'])  

#split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(final_vectorized_data, total_data[sentiment],test_size=0.3, random_state=69)

#tf-idf
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train)

from sklearn.naive_bayes import MultinomialNB,BernoulliNB   # Naive Bayes Classifier
from sklearn.metrics import accuracy_score

# #multinomial
model_naive = MultinomialNB().fit(X_train, y_train) 

predicted_naive = model_naive.predict(X_test)
score_naive = accuracy_score(predicted_naive, y_test)
print("Accuracy with Naive-bayes multinomial test: ",score_naive)

predicted_naive_train = model_naive.predict(X_train)
score_naive = accuracy_score(predicted_naive_train, y_train)
print("Accuracy with Naive-bayes multinomial train: ",score_naive)



#Bernouli
model_naive_Bernoulli = BernoulliNB().fit(X_train, y_train) 

predicted_naive_Bernoulli = model_naive_Bernoulli.predict(X_test)
score_naive_Bernoulli = accuracy_score(predicted_naive_Bernoulli, y_test)
print("Accuracy with Naive-bayes bernoulli test: ",score_naive_Bernoulli)

predicted_naive_Bernoulli = model_naive_Bernoulli.predict(X_train)
score_naive_Bernoulli = accuracy_score(predicted_naive_Bernoulli, y_train)
print("Accuracy with Naive-bayes bernoulli train: ",score_naive_Bernoulli)

# multinomial tf-idf
clf = MultinomialNB().fit(X_train_tfidf, y_train)

predicted_naive_tfidf = clf.predict(X_test)
score_naive = accuracy_score(predicted_naive_tfidf, y_test)
print("Accuracy with Naive-bayes multinomial tfidf: ",score_naive)

# predicted_naive_train_tfidf = clf.predict(X_train)
# score_naive = accuracy_score(predicted_naive_train_tfidf, y_train)
# print("Accuracy with Naive-bayes multinomial train tfidf: ",score_naive)


# 10-fold cross validation
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
cv = KFold(n_splits=10, random_state=1, shuffle=True)
model = MultinomialNB()
scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy 10-fold: %.3f (%.3f)' % (mean(scores), std(scores)))

# # 2-fold cross validation
# from numpy import mean
# from numpy import std
# from sklearn.model_selection import KFold
# from sklearn.model_selection import cross_val_score
# cv = KFold(n_splits=5, random_state=1, shuffle=True)
# model = MultinomialNB()
# scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
# print('Accuracy 2-fold: %.3f (%.3f)' % (mean(scores), std(scores)))








from sklearn.metrics import classification_report
print(classification_report(y_test, predicted_naive))



end=timer()


# In[5]:


from datetime import timedelta
print ("Execution time HH:MM:SS:",timedelta(seconds=end-start))


# In[66]:


docs_new = ['very fabulous', 'this is very awful', 'die','bye','good perfect','you are a fabulous person']
X_new_counts = count_vectorizer.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
     print('%r => %s' % (doc,category))


# In[ ]:




