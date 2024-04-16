#!/usr/bin/env python
# coding: utf-8

# In[8]:



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


total_data = pd.read_csv("file:///c:/Users/fazal/Desktop/Uni/project/dataset/twitter-airline-sentiment/Tweets.csv", encoding="ISO-8859-1")


# In[10]:


total_data=total_data[['tweet_id','airline_sentiment', 'text']]


# In[11]:


total_data = total_data.rename(columns={'airline_sentiment': 'Sentiment','tweet_id':'ItemID','text':'SentimentText'})


# In[12]:


with open('c:/Users/fazal/Desktop/Uni/project/dataset/contractions.json', 'r') as f:
    contractions_dict = json.load(f)
contractions = contractions_dict['contractions']


# In[13]:


pd.set_option('display.max_colwidth', None)


# In[14]:


total_data.head()


# In[15]:


tweet = total_data.columns.values[2]
sentiment = total_data.columns.values[1]
tweet, sentiment


# In[16]:


total_data.info()


# In[17]:


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


# In[18]:


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


# In[19]:


total_data['processed_tweet'] = np.vectorize(process_tweet)(total_data[tweet])


# In[20]:


total_data.head(10)


# In[21]:


#count vectorizer
from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer(ngram_range=(1,2))    # Unigram and Bigram
final_vectorized_data = count_vectorizer.fit_transform(total_data['processed_tweet'])  
final_vectorized_data


# In[22]:


#tf-idf vectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer 

# tf_idf_vectorizer = TfidfVectorizer(use_idf=True,ngram_range=(1,3))
# final_vectorized_data = tf_idf_vectorizer.fit_transform(total_data['processed_tweet'])

# final_vectorized_data


# In[23]:



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(final_vectorized_data, total_data[sentiment],
                                                    test_size=0.2, random_state=69)


# In[24]:


print("X_train_shape : ",X_train.shape)
print("X_test_shape : ",X_test.shape)
print("y_train_shape : ",y_train.shape)
print("y_test_shape : ",y_test.shape)


# In[25]:


from sklearn.naive_bayes import MultinomialNB,BernoulliNB   # Naive Bayes Classifier

model_naive = MultinomialNB().fit(X_train, y_train) 
predicted_naive = model_naive.predict(X_test)
model_naive_Bernoulli = BernoulliNB().fit(X_train, y_train) 
predicted_naive_Bernoulli = model_naive_Bernoulli.predict(X_test)


# In[27]:


from sklearn.metrics import confusion_matrix

plt.figure(dpi=600)
mat = confusion_matrix(y_test, predicted_naive)
sns.heatmap(mat.T, annot=True, fmt='d', cbar=False)

plt.title('Confusion Matrix for Naive Bayes')
plt.xlabel('true label')
plt.ylabel('predicted label')
# plt.savefig("assets/confusion_matrix.png")
plt.show()


# In[28]:


from sklearn.metrics import accuracy_score

score_naive = accuracy_score(predicted_naive, y_test)
print("Accuracy with Naive-bayes: ",score_naive)
score_naive_Bernoulli = accuracy_score(predicted_naive_Bernoulli, y_test)
print("Accuracy with Naive-bayes: ",score_naive_Bernoulli)


# In[29]:


from sklearn.metrics import classification_report
print(classification_report(y_test, predicted_naive))


# In[ ]:




