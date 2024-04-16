#!/usr/bin/env python
# coding: utf-8

# In[1]:


import findspark

findspark.init()


# In[2]:


from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()


# In[3]:


import pyspark
from pyspark.mllib.feature import HashingTF
# from pyspark.ml.feature import HashingTF, IDF ,Tokenizer
from pyspark.mllib.feature import IDF
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD, LogisticRegressionModel
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.feature import ChiSqSelector
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import os, tempfile
import csv
import string
import random
#import modules
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from timeit import default_timer as timer
from datetime import timedelta


# In[4]:


from pyspark import SparkContext
sc =SparkContext.getOrCreate()


# In[5]:


# Removing Stop words
def remove_values_from_list(the_list, val):
    return [value for value in the_list if value != val and value != 'i']

def cleanText(SentimentWords):
    s=SentimentWords
    for w in range(0,len(SentimentWords)):
        if('@' in SentimentWords[w]):
            s[w]='AT_USER'
        if('www.' in SentimentWords[w]):
            s[w]='URL'
        if('http://' in SentimentWords[w]):
            s[w]='URL'
        if('https://' in SentimentWords[w]):
            s[w]='URL'
        for i in range(0,len(SentimentWords[w])):
            flag=0
            if len(SentimentWords[w])>i+2 and SentimentWords[w][i]==SentimentWords[w][i+1] and SentimentWords[w][i]==SentimentWords[w][i+2]:
                for j in range(i+2,len(SentimentWords[w])):
                    if SentimentWords[w][i]==SentimentWords[w][j]:
                        flag=1
                        if len(SentimentWords[w])>j+1 and SentimentWords[w][i]!=SentimentWords[w][j+1]:
                            break
                    if j==(len(SentimentWords[w]))-1 and SentimentWords[w][i]==SentimentWords[w][j]:
                        flag=1
                        break
                if flag==1:
                    s[w]=SentimentWords[w].replace(SentimentWords[w][i:j+1],SentimentWords[w][i])
        if (SentimentWords[w]!='' and SentimentWords[w][0].isdigit()):
            s[w]=''
            
    for i in range(0,len(stop)):
        if stop[i] in SentimentWords:
#             s.append(stop[i])
            s = remove_values_from_list(s, stop[i])
    s=[word.strip(string.punctuation) for word in s]
    while '' in s:
        s.remove('')
    return s



#Compute TF
def CompTF(tshuffle_rdd):
    t_rdd=sc.parallelize([row[3] for row in tshuffle_rdd.collect()])
    hashingTF = HashingTF(25000)
    tf = hashingTF.transform(t_rdd)
    return tf

def CompTF_withNumFeatures(tshuffle_rdd):
    t_rdd=sc.parallelize([row[3] for row in tshuffle_rdd.collect()])
    hashingTF = HashingTF(1500)
    tf = hashingTF.transform(t_rdd)
    return tf

#Compute IDF
def CompIDF(tf):
    tf.cache()
    idf = IDF().fit(tf)
    return idf

#Compute TFIDF
def CompTFIDF(tf,idf):
    tfidf = idf.transform(tf)
    return tfidf

#Feature Extraction
def Convert_to_LabeledPoint(labels,features):
    training = labels.zip(features).map(lambda x: LabeledPoint(x[0], x[1]))
    return training

#Training - NB
def NB_train(training):
    model = NaiveBayes.train(training,1.0)
    return model

#Testing
def test(model,labels,features):
    labels_and_preds = labels.zip(model.predict(features)).map(
                                lambda x: {"actual": float(x[0]), "predicted": float(x[1])})
    acc=100.0*((labels_and_preds.filter(lambda x:x["actual"]==x["predicted"]).count()) / (labels.count()))
    return acc

#Final Test

def test_final(model,labels,features):
    labels_and_preds = labels.zip(model.predict(features)).map(
                                lambda x: {"actual": float(x[0]), "predicted": float(x[1])})
    acc=100.0*((labels_and_preds.filter(lambda x:x["actual"]==x["predicted"]).count()) / (labels.count()))
    return (labels_and_preds,acc)


# In[16]:


#begin
start=timer()
# df = spark.read.csv('file:///c:/Users/fazal/Desktop/Uni/project/dataset/twitter-airline-sentiment/Tweets.csv',header=True)
# df=df.withColumnRenamed("airline_sentiment","sentiment")
df = spark.read.csv('file:///c:/Users/fazal/Desktop/Uni/project/dataset/trainingandtestdata/training.1600000.processed.noemoticon.csv',header=True)
df=df.union(df)
df=df.union(df)
df=df.union(df)
df=df.withColumnRenamed(df.columns[0],"sentiment")
df=df.withColumnRenamed(df.columns[5],"text")
df1=df.select('sentiment','text')
# df1.show(1000)

# only show tweets with neutral and positive and negative sentiment and remove the others
df_filtered=df1.filter((df1['sentiment']=='4') | (df1['sentiment']=='0')| (df1['sentiment']=='2') )
numTransactionEachSentiment = df_filtered.groupBy("sentiment").count()
# numTransactionEachSentiment.show(5)
# df_filtered.show(20)

# drop duplicated data
# df_dropped_duplicated=df_filtered.dropDuplicates()
df_dropped_duplicated=df_filtered.na.drop()
# df_dropped_duplicated.show(3)

########## convert column to lower case in pyspark
from pyspark.sql.functions import lower, col
df_lower_case=df_dropped_duplicated.select("sentiment", (lower(col('text'))).alias('text'))
# df_lower_case.show(5)

#divide data, 70% for training, 30% for testing
dividedData = df_lower_case.randomSplit([0.7, 0.3,0.0]) 
trainingData = dividedData[0] #index 0 = data training
testingData = dividedData[1] #index 1 = data testing
extraForNow = dividedData[2] #index 1 = data testing
train_rows = trainingData.count()
test_rows = testingData.count()
print ("Training data rows:", train_rows, "; Testing data rows:", test_rows)

#seperate words
tokenizer = Tokenizer(inputCol="text", outputCol="SentimentWords")
tokenizedTrain = tokenizer.transform(trainingData)
tokenizedTest = tokenizer.transform(testingData)
# tokenizedTrain.show(truncate=False, n=5)
# tokenizedTest.show(truncate=False, n=5)


#clean data and remove stop words
stop=['i','me', 'my', 'myself', 'we', 'our', 'ours', 'â€œ','ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
row=tokenizedTrain.first()
import pyspark.sql.functions as F
from pyspark.sql.types import *

#convert to a UDF Function by passing in the function and return type of function
cleanTextUDF = F.udf(cleanText,ArrayType(StringType()))
trainCleaned = tokenizedTrain.withColumn("SentimentWordsCleaned", cleanTextUDF('SentimentWords'))
testCleaned = tokenizedTest.withColumn("SentimentWordsCleaned", cleanTextUDF('SentimentWords'))
# trainCleaned.show(10)
# testCleaned.show(10)

#add label column
from pyspark.sql import functions as f
trainCleaned_labelAdded=trainCleaned.withColumn('label', f.when(f.col('sentiment') == "0", 0).otherwise(when(f.col('sentiment') == "4", 2).otherwise(1)))
testCleaned_labelAdded=testCleaned.withColumn('label', f.when(f.col('sentiment') == "0", 0).otherwise(when(f.col('sentiment') == "4", 2).otherwise(1)))
# trainCleaned_labelAdded.show(5)
# testCleaned_labelAdded.show(5)

#Accuracy of Training data using train.csv itself using NB
rdd_train = trainCleaned_labelAdded.rdd.map(list)
rdd_test = testCleaned_labelAdded.rdd.map(list)
tf_train=CompTF(rdd_train)
idf_train=CompIDF(tf_train)
tfidf_train=CompTFIDF(tf_train,idf_train)
# print(tfidf_train.first())

training_NB = Convert_to_LabeledPoint(sc.parallelize([row[4] for row in rdd_train.collect()]),tfidf_train)
# print(training_NB.first())

#training accuracy for NB ML technique - training with training data
model_train_NB=NB_train(training_NB)

#testing NB with training data 
accuracy_NB=test(model_train_NB,sc.parallelize([row[4] for row in rdd_train.collect()]),tfidf_train)
print ("TRAINING ACCURACY:-\n")
print("The accuracy for the training dataset tested on the training data itself using NB is",accuracy_NB,"%")
print ("\n")

# # KFold NB

# print ("10-FOLD CV ACCURACIES FOR ALL ITERATIONS\n")

p=rdd_train.randomSplit(weights=[0.5,0.5], seed=1)

tot_NB_kfold=0
NB_kfold_set=[]
for i in range(0,len(p)):
    test_RDD=p[i]
    train_tempRDD=sc.emptyRDD()
    for j in range(0,len(p)):
        if i!=j:
            train_tempRDD=train_tempRDD.union(p[j])
    tf_train=CompTF(train_tempRDD)
    idf_train=CompIDF(tf_train)
    tfidf_train=CompTFIDF(tf_train,idf_train)
    training = Convert_to_LabeledPoint(sc.parallelize([row[4] for row in train_tempRDD.collect()]),tfidf_train)
    model_train=NB_train(training)
    tf_test=CompTF(test_RDD)
    tfidf_test=CompTFIDF(tf_test,idf_train)
    accuracy=test(model_train,sc.parallelize([row[4] for row in test_RDD.collect()]),tfidf_test)
    print ("The accuracy for number",i+1,"kth partition test for 10-fold cross validation for NB is",accuracy,"%")
    NB_kfold_set.append(accuracy)
    tot_NB_kfold=tot_NB_kfold+accuracy
avg_NB_kfold=tot_NB_kfold/len(p)
print ("\n")
print ("The average accuracy for NB after 10-fold cross validation is",avg_NB_kfold,"%")
print ("\n")

tf_test=CompTF(rdd_test)
tf_train=CompTF(rdd_train)
idf_train=CompIDF(tf_train)
tfidf_test=CompTFIDF(tf_test,idf_train)

labels_and_preds_NB,accu_NB = test_final(model_train_NB,sc.parallelize([row[4] for row in rdd_test.collect()]),tfidf_test)
metrics2 = BinaryClassificationMetrics(labels_and_preds_NB.map(lambda x: (x["predicted"], x["actual"])))
print ("\nTEST ACCURACY:-\n")
print("The accuracy of prediction for NB on testing data is",accu_NB,"%")
# objects = ('Training Accuracy', '10-Fold CV', 'Testing Accuracy')
# y_pos = np.arange(len(objects))
# performance = [accuracy_NB,avg_NB_kfold,accu_NB]
# plt.bar(y_pos, performance, align='center', alpha=0.5)
# plt.xticks(y_pos, objects)
# plt.xlabel('Classifications')
# plt.ylabel('Accuracy')
# plt.title('Naive Bayes Classifier - Accuracies')
 
# plt.show()

# metrics = MulticlassMetrics(labels_and_preds_NB.map(lambda x: (x["predicted"], x["actual"])))

# # Overall statistics
# print("\nSummary Stats_NB\n")
# # Statistics by class
# labels = (sc.parallelize([row[4] for row in rdd_test.collect()])).distinct().collect()
# for label in sorted(labels):
#     print("Class %s precision_NB = %s" % (label, metrics.precision(label)))
#     print("Class %s recall_NB = %s" % (label, metrics.recall(label)))
#     print("Class %s F1 Measure_NB = %s" % (label, metrics.fMeasure(float(label), beta=1.0)))

# # Weighted stats
# print("\nAvg/Weighted recall_NB = %s" % metrics.weightedRecall)
# print("Avg/Weighted precision_NB = %s" % metrics.weightedPrecision)
# print("Avg/Weighted F(1) Score_NB = %s" % metrics.weightedFMeasure())

# cm=metrics.confusionMatrix().toArray()
# print("\nConfusion matrix_NB=")
# print(cm)
# print("\n")
# metrics2 = BinaryClassificationMetrics(labels_and_preds_NB.map(lambda x: (x["predicted"], x["actual"])))

# # Area under ROC curve
# print("Area under ROC_NB = %s" % metrics2.areaUnderROC)

end=timer()


# In[14]:


print ("Execution time HH:MM:SS:",timedelta(seconds=end-start))


# In[ ]:




