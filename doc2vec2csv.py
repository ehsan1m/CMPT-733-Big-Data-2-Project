
#This is script is used to generate the doc2vec vectors and save them as a CSV file

import numpy as np
import pandas as pd
import re
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedLineDocument
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE 
#from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split



filename = 'Data/mergedDataSet.csv' #The complete file
df = pd.read_csv(filename)
#df = pd.read_csv(filename).iloc[0:1000]

#Removing the stop words
stop = open('stop_words.txt','r').read().split()
def RemoveStopWords(row):
    row = row.lower().split() # converting to lower case and splitting
    str1 = ''
    for item in row:
        if item not in stop: #removing stop words
            item = re.sub(r'[^\w\s]','',item) #removing punctutions
            str1 += (item + ' ')
    return str1
target = df['merged_rating'].values #Saving the target variable as a numpy array
df = df['comment_text'].apply(RemoveStopWords)


#Saving to CSV so that TaggedLineDocument can read it. TaggedLineDocument is needed for the doc2vec model.
commentsFileName = 'Data/comments.csv'
df.to_csv(commentsFileName,index=False) # Writing the comments to a CSV file to be read by TaggedLineDocument next
documents = TaggedLineDocument(commentsFileName) # Tags each sentence (0,1,2,...)

modelSize = 500 # Will represent each comment with a vector of size 500
modelWindow = 8 
model = Doc2Vec(documents, vector_size=modelSize, window=modelWindow, min_count=1, workers=4, dbow_words=1, epochs=20)

#Creating a numpy array to keep the data and to be used by a machine learning model as the feature vector
data = np.zeros((len(model.docvecs),len(model.docvecs[0])))
for i in range(len(model.docvecs)):
    data[i]=model.docvecs[i]


#Creating validation sets
X_train, X_val, y_train, y_val = train_test_split(data,target, test_size=0.2, random_state=0)

#Upsamlping the training data
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_sample(X_train, y_train)

dfTrain = pd.DataFrame(data=X_res,dtype=float)
dfTrain['label']=y_res
dfTrain.to_csv('Data/FeaturizedDataTraining.csv')
#dfTrain.to_csv('Data/FeaturizedDataTrainingSample.csv')
dfVal = pd.DataFrame(data=X_val,dtype=float)
dfVal['label']=y_val
dfVal.to_csv('Data/FeaturizedDataValidation.csv')
#dfVal.to_csv('Data/FeaturizedDataValidationSample.csv')


print ('finished')

