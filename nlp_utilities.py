# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 16:05:53 2020

@author: Derek G. Nokes
"""

import pandas as pd
import numpy as np
import requests
import datetime
from bs4 import BeautifulSoup

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import nltk
import random


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import collections
import itertools
#from nltk.corpus import stopwords, reuters
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

from nltk import metrics
from nltk.classify.util import accuracy

from nltk.classify import NaiveBayesClassifier
from nltk.classify import DecisionTreeClassifier
from nltk.classify import MaxentClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from nltk.classify import util, ClassifierI, MultiClassifierI

# define function to convert raw text to words, excluding
# stopwords
def convert2wordsExcludeStopwords(rawText):
    # tokenize raw text
    tokens = nltk.word_tokenize(rawText)
    # extract only alphabetic
    words = [w for w in tokens if w.isalpha()]
    # convert to lowercase
    words = [w.lower() for w in words]
    # fetch stopword list
    stopwordList = nltk.corpus.stopwords.words('english') 
    # remove stop words
    words = [w for w in words if w not in stopwordList]
    # count number of words
    nWords=len(set(words))
    # return words and number of words
    return words,nWords

# define function to convert raw text to words, excluding
# stopwords, then apply stemming
def convert2StemmedWordsExcludeStopwords(rawText):
    # convert raw text to words excluding stopwords
    words,nWords=convert2wordsExcludeStopwords(rawText)
    # define stemmer
    ps = nltk.stem.porter.PorterStemmer()
    # apply stemming
    wordsStemmed = [ps.stem(w) for w in words]
    # determine set
    wordsStemmedSet = set(wordsStemmed)
    # compute number of words after stemming
    nWordsStemmed=len(wordsStemmedSet)

    return wordsStemmed,nWordsStemmed,nWords

# define function to convert raw text to words, excluding
# stopwords, then apply lemmatization
def convert2LemmatizedWordsExcludeStopwords(rawText):
    # convert raw text to words excluding stopwords
    words,nWords=convert2wordsExcludeStopwords(rawText)
    # initialize
    wordnet = nltk.stem.WordNetLemmatizer()
    # apply lemmatization
    wordsLemmatized = [wordnet.lemmatize(w) for w in words]
    # determine set
    wordsLemmatizedSet = set(wordsLemmatized)
    # compute number of words lemmatized
    nWordsLemmatized=len(wordsLemmatizedSet)
    # display number of words lemmatized
    nWordsLemmatized    
    
    return wordsLemmatized,nWordsLemmatized,nWords

def addWordsExcludeStopwords(df,inputField,outputField='words'):
    # define function to convert raw text of document to list
    # of alphabetic words (excluding stopwords)
    fWords = lambda d : convert2wordsExcludeStopwords(d)[0]
    # apply function to extract words
    df[outputField]=df[inputField].apply(fWords)
    
    return df 

def addStemmedWordsExcludeStopwords(df,inputField,outputField='wordsStemmed'):
    # define function to convert raw text of document to list
    # of alphabetic words (excluding stopwords)
    fWordsStemmed = lambda d : convert2StemmedWordsExcludeStopwords(d)[0]
    # apply function to extract words
    df[outputField]=df[inputField].apply(fWordsStemmed)
    
    return df 
    
def addLemmatizedWordsExcludeStopwords(df,inputField,outputField='wordsLemmatized'):
    # define function to convert raw text of document to list
    # of alphabetic words (excluding stopwords)
    fWordsLemmatized = lambda d : convert2LemmatizedWordsExcludeStopwords(d)[0]
    # apply function to extract words
    df[outputField]=df[inputField].apply(fWordsLemmatized)
    
    return df

def addNWords(df,inputField='words',outputFields='nWords'):
    # add word count
    df[outputFields]=df[inputField].apply( lambda d : len(d))

    return df

def addNSetWords(df,inputField='words',outputFields='nSetWords'):
    # add word set count
    df[outputFields]=df[inputField].apply( lambda d : len(set(d)))

    return df

def addNWordsStemmed(df,inputField='wordsStemmed',outputFields='nWordsStemmed'):
    # add stemmed word count
    df=addNWords(df,inputField,outputFields)

    return df

def addNSetWordsStemmed(df,inputField='wordsStemmed',outputFields='nSetWordsStemmed'):
    # add stemmed set word count
    df=addNSetWords(df,inputField,outputFields)

    return df

def addNWordsLemmatized(df,inputField='wordsLemmatized',outputFields='nWordsLemmatized'):
    # add lemmatized word count
    df=addNWords(df,inputField,outputFields)

    return df

def addNSetWordsLemmatized(df,inputField='wordsLemmatized',outputFields='nSetWordsLemmatized'):
    # add lemmatized set word count
    df=addNSetWords(df,inputField,outputFields)

    return df

# randomize order of full data set to ensure that deterministic ordering of 
# data does not impact results
def shuffleDf(df,randomState=1234567890):
    # set random state
    prng = np.random.RandomState(randomState)
    # use random state to create index to reorder dataframe
    shuffleIndex=prng.permutation(df.index)
    # apply index to reorder dataframe
    df=df.loc[shuffleIndex]
    # reset the index and add a column with the old index
    df.reset_index(drop=False,inplace=True)
    
    return df

# covert words to bag of words
def bagOfWords(words):
    return dict([(w, True) for w in words])

# covert words to bag of words, excluding list of words
def bagOfWordsNotInSet(words, excludeWords):
    return bagOfWords(set(words) - set(excludeWords))

# covert words to bag of words + bigrams
def bagOfBigramsWords(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigramFinder = BigramCollocationFinder.from_words(words)
    bigrams = bigramFinder.nbest(score_fn, n)
    return bagOfWords(words + bigrams)

def bagOfWordsInSet(words, includeWords):
    return bagOfWords(set(words) & set(includeWords))

def applyBagOfWords(df,inputField='wordsLemmatized',outputField='wordsLemmatizedBOW'):
    # define function to apply bag of words to each document
    fBagOfWords = lambda d : bagOfWords(d)
    # apply bag of words to each document
    df[outputField]=df[inputField].apply(fBagOfWords)    
    
    return df

def applyBagOfBigramWords(df,inputField='wordsLemmatized',outputField='wordsLemmatizedBOBW'):
    # define function to apply bag of words to each document 
    # and include bigrams
    fBagOfBigramWords=lambda d : bagOfBigramsWords(d)
    # apply bag of words + bigrams to each document 
    df[outputField]=df[inputField].apply(fBagOfBigramWords)
        
    return df


def applyBagOfWordsNotInSet(df,excludeWords=[],inputField='wordsLemmatized',outputField='wordsLemmatizedBOWNIS'):
    # define function to apply bag of words excluding certain words
    # to each document
    fBagOfWordsNotInSet = lambda d : bagOfWordsNotInSet(d, excludeWords)
    # apply bag of words to each document
    df[outputField]=df[inputField].apply(fBagOfWordsNotInSet)    
    
    return df

def mapBOWFeatureSetAndClassification2ListOfTuples(df,classificationField='classification',outputField='wordsLemmatizedBOW'):
    # map bag of words feature set and classification into list of tuples
    featureSetBOW=list(df[[outputField,
    classificationField]].itertuples(index=False, name=None))
    
    return featureSetBOW
    
def mapBOBWFeatureSetAndClassification2ListOfTuples(df,classificationField='classification',outputField='wordsLemmatizedBOBW'):
    # map bag of words + bigrams feature set and classification into 
    # list of tuples
    featureSetBOBW=mapBOWFeatureSetAndClassification2ListOfTuples(df,classificationField,outputField)
    
    return featureSetBOBW

def buildWordClassificationDataFrame(df,classificationField='classification',wordListType='wordsLemmatized'):

    masterWords=list()
    masterClasses=list()

    # iterate over each document in corpus
    for rowIndex, row  in df.iterrows():
        # extract class and add to master word list
        [masterWords.append(w) for w in df[wordListType].loc[rowIndex]]
        # extract class and add to master class list
        [masterClasses.append(df[classificationField].loc[rowIndex]) for w in df[wordListType].loc[rowIndex]]
 
    # create dataframe with words and corresponding sector
    wordClassification=pd.DataFrame(list(zip(masterWords, masterClasses)),
        columns=['word',classificationField])

    return wordClassification,masterWords,masterClasses

def groupByWordAndClassification(wordClassification,inputField='word',classificationField='classification'):
    # group by word and classification
    wordByClassification=wordClassification.groupby([inputField,classificationField])[inputField].count()
    # unstack to word x sector matrix
    wordByClassificationFrequency=wordByClassification.unstack()
    # fill NaN with zero
    wordByClassificationFrequency.fillna(value=0,inplace=True)
    
    return wordByClassificationFrequency

def convertWordByClassificationFrequency2Percent(wordByClassificationFrequency):
    # compute column totals
    columnTotals=wordByClassificationFrequency.sum(axis=0)
    # compute row totals
    rowTotals=wordByClassificationFrequency.sum(axis=1)
    # convert frequency count to frequency percent 
    wordByClassificationFrequencyPercent=wordByClassificationFrequency/columnTotals

    return  wordByClassificationFrequencyPercent,columnTotals,rowTotals

def convertFrequencyPercent2Flag(wordByClassificationFrequencyPercent):
    # convert frequency percent to True/False
    wordInClassification=wordByClassificationFrequencyPercent>0.0
    
    return wordInClassification

def plotWordByClassificationFrequencyPercent(wordByClassificationFrequencyPercent,classLabel,topN=30,figureSize=(10,3)):
    
    wordFrequencyForClassLabel=wordByClassificationFrequencyPercent[classLabel].sort_values(ascending=False)
    plt.figure(figsize=figureSize)
    wordFrequencyForClassLabel[1:topN].plot(kind='bar')
    plt.title(classLabel)
    plt.ylabel("Class Frequency (%)")
    plt.xlabel("Word")
    
    return

def findMostCommonWordsInAllClassifications(wordByClassificationFrequency):
    # convert frequency count to frequency percent
    wordByClassificationFrequencyPercent,columnTotals,rowTotals=convertWordByClassificationFrequency2Percent(wordByClassificationFrequency)
    # convert frequency percent to True/False
    wordInClassification=convertFrequencyPercent2Flag(wordByClassificationFrequencyPercent)
    # compute row totals
    flagRowTotals=wordInClassification.sum(axis=1)

    # compute corpus total
    nWordsCorpus=rowTotals.sum(axis=0)
    # compute corpus all words by frequency
    allWordsByFrequency=(rowTotals/nWordsCorpus)

    # join high frequency words with number of classifications in which they appear
    commonWordsInAllClassifications=pd.concat([allWordsByFrequency,
        flagRowTotals], axis=1, join='inner', ignore_index=False)
    # define column names
    columnNames={0 : 'wordFrequencyPercent', 1 : 'nClassifications'}
    # rename column names
    commonWordsInAllClassifications.rename(columns=columnNames,inplace=True)
    # sort
    mostCommonWordsInAllClassifications=commonWordsInAllClassifications.sort_values(by=['wordFrequencyPercent',
        'nClassifications'],ascending=False)
    
    return mostCommonWordsInAllClassifications

def buildFeatureSetsBySplitBOW(dfDict,classificationField='classification',wordType='wordsLemmatizedBOW'):

    dfTrain=dfDict['train']
    dfDevTest=dfDict['dev_test']
    dfTest=dfDict['test']
    
    # map bag of words + bigrams feature set and classification into list of tuples (train)
    trainFeatureSet=mapBOWFeatureSetAndClassification2ListOfTuples(dfTrain,
        classificationField,wordType)
    # map bag of words + bigrams feature set and classification into list of tuples (dev test)
    devTestFeatureSet=mapBOWFeatureSetAndClassification2ListOfTuples(dfDevTest,
        classificationField,wordType)
    # map bag of words + bigrams feature set and classification into list of tuples (test)
    testFeatureSet=mapBOWFeatureSetAndClassification2ListOfTuples(dfTest,
        classificationField,wordType)

    featureSetDict=dict()
    featureSetDict['train']=trainFeatureSet
    featureSetDict['dev_test']=devTestFeatureSet
    featureSetDict['test']=testFeatureSet
    
    return featureSetDict

def accuracyBySplit(classifier,featureSetDict):
    accuracyDict=dict()

    # compute accuracy for each split
    for splitLabel in featureSetDict:
        accuracyDict[splitLabel]=nltk.classify.accuracy(classifier, featureSetDict[splitLabel])
        
    return accuracyDict

def addClassBySplit(classifier,dfDict,featureSetDict,predictedClassField='predicted_class',classificationField='classification',wordType='wordsLemmatizedBOW',displayDigits=4):

    # create predicted classes (train)
    fClassify=lambda d : classifier.classify(d)
    
    output=dict()
    predictionsDict=dict()
    confusionMatrixDict=dict()
    classificationReportDict=dict()

    # compute accuracy for each split
    for splitLabel in featureSetDict:
        # copy data for split
        predictionsDf=dfDict[splitLabel].copy()
        # add class predicted by classifier for split
        predictionsDf[predictedClassField]=predictionsDf[wordType].apply(fClassify)
        # store predictions for split
        
        # covert predicted class column of dataframe to list
        predictionsList=predictionsDf[predictedClassField].tolist()
        # extract actual classes (ground truth)
        groundTruthList = [g  for (n, g) in featureSetDict[splitLabel]]
        # create confusion matrix
        confusionMatrix=confusion_matrix(groundTruthList, predictionsList)
        # get unique classes (for split)
        classLabels=list(predictionsDf[classificationField].sort_values().unique())
        # create table with precision, recall, f1-score, and support (for split)
        classificationReport=classification_report(groundTruthList, predictionsList,
            digits=displayDigits,output_dict=True)
        # build confusion matrix and classification report dataframes (for split)
        confusionMatrixDf=pd.DataFrame(index=classLabels,data=confusionMatrix,columns=classLabels)
        classificationReportDf=pd.DataFrame(classificationReport).T
        # store output for split
        predictionsDict[splitLabel]=predictionsDf
        confusionMatrixDict[splitLabel]=confusionMatrixDf
        classificationReportDict[splitLabel]=classificationReportDf
    
    # store output for all splits
    output['predictionsDict']=predictionsDict
    output['confusionMatrixDict']=confusionMatrixDict
    output['classificationReportDict']=classificationReportDict
        
    return output