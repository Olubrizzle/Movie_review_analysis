#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:01:12 2019

@author: oluwolealowolodu
"""

#Natural language processing...importing modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import re
from bs4 import BeautifulSoup
import tensorflow as tf
#read in file
train = pd.read_csv('labeledTrainData.tsv', header=0, delimiter='\t', quoting=3 )


nltk.download() #download textdata set including stop words


#getting stop words with nltk
from nltk.corpus import stopwords

def review_to_words(raw_review):
    bs = BeautifulSoup(raw_review).get_text()
    letters_only = re.sub('[^a-zA-Z]', ' ', bs)
    #tokenization
    words_only = letters_only.lower().split()
    stops = set(stopwords.words('english'))
    words = [w for w in words_only if not w in stops]
    return (" ".join(words))

num_review = train['review'].size #5000

#initiate empty list to hold the clean reviews
clean_train_review = []

for i in xrange(0, num_review):
    clean_train_review.append(review_to_words(train['review'][i]))


'''creating bag of words'''
from sklearn.feature_extraction.text import CountVectorizer
# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 
train_data_features = vectorizer.fit_transform(clean_train_review)
#convert result to array, cos of their ease to work with
train_data_features = train_data_features.toarray()
print train_data_features.shape

#vocab = vectorizer.get_feature_names()
#print vocab


#training using random forest model and gradientboosting
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
#Using Grid search method for hyperparameter tuning
 #=============================================================================
rfc=RandomForestClassifier(random_state=2)
param_grid = {"max_features": ['auto', 'sqrt', 'log2'],
              "min_samples_split": [5, 10, 50, 100],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
 
grid = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=10, n_jobs=-1, iid=False)
grid.fit(train_data_features,train['sentiment'])
grid.best_params_

# =============================================================================
# #initializing with 120 trees
# forest = RandomForestClassifier(n_estimators=100)
# #fit the model to the training set using bag of words
# forest = forest.fit(train_data_features, train['sentiment'])
# =============================================================================

#model hyperparameters
rfc = RandomForestClassifier(bootstrap = False, criterion = 'entropy', max_features = 'auto',
                           min_samples_split = 100, n_estimators = 100)

#fit the model to the training set using bag of words
rfc = rfc.fit(train_data_features, train['sentiment'])

#.....................cross validation to split train-test then root mean squared error to check accuracy
mse = cross_val_score(rfc, train_data_features, train['sentiment'], cv=10, scoring = 'neg_mean_squared_error').mean()
rmse = (mse*-1)**.5
strrfc = str(rfc)
end_index = strrfc.index('(')
print('root MSE of {} : '.format(strrfc[:end_index]) + str(round(rmse,6)))

#ESTIMATING R-SQUARED
R2 = rfc.score(train_data_features, train['sentiment'])
strmodel = str(rfc)
end_index = strrfc.index('(')
print('R-squared of {} : '.format(strrfc[:end_index]) + str(round(R2,6)))

#reading test file 
test = pd.read_csv('testData.tsv', delimiter='\t', header=0, quoting=3)
print test.shape #2500, 2
#appending the review to a list
num2_reviews = len(test['review'])
clean_test_review = []
for i in xrange(0, num2_reviews):
    clean_test_review.append(review_to_words(test['review'][i]))

#converting the bag of words of test to numpy array
test_data_features = vectorizer.transform(clean_test_review)
test_data_features = test_data_features.toarray()

#predicting with random forest
result = rfc.predict(test_data_features)

#creating csv file for kaggle sumssion
subm1 = pd.DataFrame(data={'id':test['id'], 'sentiment':result})
subm1.to_csv('subm1nlp.csv', index=False, quoting=3)

'''Public Score 0.84760 inposition 344 on first submision'''

import matplotlib.pyplot as plt

sub1 = pd.read_csv('subm1nlp.csv')
ax = sub1["sentiment"].value_counts().plot(kind='bar', label='Positive Review')
ax.set_xlabel('Class of Predicted Reviews')
ax.set_ylabel('Number of Predicted Reviews')
ax.set_title('Analysis of Movie Review by Random Forest Model')
plt.show()
R_Positive_review = [1 for i in sub1.sentiment if i == 1]
R_Negative_review = [1 for i in sub1.sentiment if i == 0]
len(R_Positive_review), len(R_Negative_review)


