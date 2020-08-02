# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 20:12:20 2020

@author: ANKUR
"""

from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
import numpy
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans
from scipy.spatial import distance
import math
import numpy as np
import sklearn.metrics.pairwise
from sklearn.metrics.pairwise import cosine_similarity
import statistics
from statistics import mean




f=open("C:\\Users\\ANKUR\\Desktop\\FINAL_YR_PROJECT\\FINAL_PROJECT\\PEARSON\\ankur.txt", 'r', errors='ignore')
#C:\\Users\\ANKUR\\MEDLINE(ORIGINAL,CONCEPTS)\\megha.txt"
punct = '''!()-[]{};:'\,<>/=?"@#$%^&*_~'''
stop_words = set(stopwords.words('english'))

#taking all sentences in an array
arr=[]
for line in f:
    for i in line.split(". "):
        arr.append(i)
f.close()

'''
#printing the lines
for i in arr:
    print(i)
    print(arr.index(i))
    print("-----------------------------------------------\n")

'''

'''
#taking all words in a single array
wordvocab=[]
for each_sentence in arr:
    for each_word in each_sentence.split():
        wordvocab.append(each_word)

#removng punctuation except "."    
punctremoved=[]
for a in wordvocab:
    c=""
    for b in a:
        if b not in punct:
            c= c+b
    punctremoved.append(c.lower())        
'''


#other way---> first removing the punctuation from the sentences
removed=[]
for each in arr:
    x=""
    for i in each:
        if i not in punct:
            x=x+i
    removed.append(x.lower())        

#removing stopwords
filtered_sentences=[]
for sentence in removed:
    #print(sentence)
    filtered=[]
    for word in sentence.split():
        if not word in stop_words:
            filtered.append(word)
    filtered_sentences.append(filtered)        

#lemmatizing:
train_vocab=[]
lemmatizer=WordNetLemmatizer()
for each_list in filtered_sentences:
    vocabulary=[]
    for each_word in each_list:
        vocabulary.append(lemmatizer.lemmatize(each_word))
    train_vocab.append(vocabulary)   
    
#word to vector conversion
from gensim.models import Word2Vec
model= Word2Vec(train_vocab, min_count=1)    #train model
print("The model is :\n", (model))   #summarize the loaded model     
dim=len(model[word])
all_words = list(model.wv.vocab)          #summarize vocabulary

#access vector for one word
#print(model['oral'])
#reference: https://machinelearningmastery.com/develop-word-embeddings-python-gensim/


#converting each sentence in vector form
summation=[]
vectorized=[]
for sentence in train_vocab:             
   # sum=np.zeros(100)
    sum=np.zeros(dim)
    for word in sentence:
        sum= sum + model[word]
    avg= sum/len(sentence)
    summation.append(sum)
    vectorized.append(avg)    

'''       
#writing into csv file:
import csv
with open('cluster.csv','wb') as file:
    for i in range(0, len(vectorized)):
        np.savetxt(file, [vectorized[i]],delimiter= ',', fmt='%f')
'''

#applying k-means
freq=[0]* len(train_vocab)
k=math.ceil(0.4 * len(arr))
#k=7
x= np.array(vectorized)

for t in range(0, 50):
    
    kmeans= KMeans(n_clusters= k)
    kmeans.fit(x)

    centroids= kmeans.cluster_centers_
    labels= kmeans.labels_

   # print ("centroids are:   ", centroids)
    #print("labels are:   ", labels)

#forming lists of clusters:
    all_clusters=[]
    for i in range(0, kmeans.n_clusters):
        cluster=[]
        counter=0
        for j in range(len(x)):
            if labels[j]==i:
                cluster.append(x[j])
           # cluster.append(vectorized[counter])
       # counter=counter+1 
        all_clusters.append(cluster)        
#print(all_clusters[0][0])
    #print(len(all_clusters))
    #print(len(all_clusters[0]))
    #print(len(all_clusters[1]))
    #print(len(all_clusters[2]))


#calculating the distances of the vectors(points) from the cluster centroids:
    represent=[]
#for i in range(0, kmeans.n_clusters):
    for i in range(len(all_clusters)):
    #len(all_clusters):
   # s=[]
   # r=[]
       s=numpy.array(centroids[i])
   # min=distance.euclidean(all_clusters[i][0], centroids[i])
       r=numpy.array(all_clusters[i][0])
       sum1=0
       sum2=0
       m_r=mean(r)
       m_s=mean(s)
       for u in range(0, dim):
           sum1+=(r[u]-m_r)*(s[u]-m_s)
           sum2+=math.sqrt((r[u]-m_r)**2) * math.sqrt((s[u]-m_s)**2)
       max=sum1/sum2    
       ind=0
       p=len(all_clusters[i])
       for j in range(1, p):
           r=numpy.array(all_clusters[i][j])
           m_r=mean(r)
           sum1=0
           sum2=0
           for u in range(0, dim):
               sum1+=(r[u]-m_r)*(s[u]-m_s)
               sum2+=math.sqrt((r[u]-m_r)**2) * math.sqrt((s[u]-m_s)**2)
       
           d=sum1/sum2
           if d > max:
               max=d
               ind=j
   #rep.append(all_clusters[i][ind])    
       represent.append(ind)
  #print(represent)
      
      #summary of each iteration:
    sentence=[]
    for i in range(0, len(represent)):
        count=-1
        for j in range(0, len(labels)):
            if i == labels[j]:
                count+=1
                if count == represent[i]:
                    sentence.append(i)
                    break
    
#sentence.sort()
    for i in range(0, len(sentence)):
        freq[sentence[i]]+= 1

#final summary sentence index in sum_sent
sum_sent=[]
for i in range(0, len(train_vocab)):
    sum_sent.append(i)
for i in range(1, len(train_vocab)-1):
    for j in range(0, len(train_vocab)-i):
        if freq[j] < freq[j+1]:
            freq[j], freq[j+1] = freq[j+1], freq[j]
            sum_sent[j], sum_sent[j+1] = sum_sent[j+1], sum_sent[j]

final_sent=[]
for i in range(0, k):
    p=sum_sent[i]
    final_sent.append(p)
final_sent.sort()
            
f=open("C:\\Users\\ANKUR\\Desktop\\FINAL_YR_PROJECT\\FINAL_PROJECT\\PEARSON\\pearson_summary.txt","w+")
c=0
for i in range(0, len(arr)):
    if i==final_sent[c]:
        f.write(arr[i])
        f.write(".\n")
        #print(arr[i])
        c+=1
        if c>=len(final_sent):
            break
f.close()