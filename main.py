#!/usr/bin/env python

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import cross_val_score,cross_val_predict
from statistics import mean
from sklearn.model_selection import GridSearchCV
import numpy as np
import random
import features
#from sklearn.base import TransformerMixin
import string
import pickle
import sys

def open_corpus(pickle_path):
	print("\n##### Reading corpus...")
	print("Reading file:", pickle_path)
	x = list()
	y = list()
	with open(pickle_path, 'rb') as annotated_pickle:
		annotated_tweets = pickle.load(annotated_pickle)
		for tweet,annotation in annotated_tweets.items():
			x.append(tweet)
			y.append(annotation)
	c = list(zip(x, y))
	c = sorted(c)
	random.seed(85)
	random.shuffle(c)
	x,y = zip(*c)
	X_train, Y_train = list(x)[:800], list(y)[:800]
	X_test, Y_test = list(x)[800:], list(y)[800:]
	return X_train, X_test, Y_train, Y_test


def evaluate(Y_guess, Y_test):
	cm = np.zeros((3, 3), dtype=int)
	np.add.at(cm, [Y_test, Y_guess], 1)
	false_p_neutral = cm[1,0]+cm[2,0]
	false_n_neutral = cm[0,1] + cm[0,2]
	false_p_favor = cm[0,1]+cm[2,1]
	false_n_favor = cm[1,0] + cm[1,2]
	false_p_against = cm[0,2]+cm[1,2]
	false_n_against = cm[2,0] + cm[2,1]
	print("Neutral: ", false_p_neutral,false_n_neutral)
	print("Favor: ", false_p_favor,false_n_favor)
	print("Against: ", false_p_against,false_n_against)
	return(cm)

def runSVC(X_train,Y_train, feature):
	stop_set = set(stopwords.words("dutch"))
	if feature == "tfidf":
		#vectorizer = TfidfVectorizer(max_features=2500, ngram_range = (1,3))#, stop_words = stop_set)
		vectorizer = FeatureUnion([('tfidf_w',TfidfVectorizer(max_features=2500, ngram_range = (1,3))),('tfidf_c',TfidfVectorizer(max_features=2500, ngram_range = (2,5), analyzer = 'char'))])
	elif feature == "embeddings":
		embeddings_pickle = open("vectors-320.pickle","rb")
		embeddings = pickle.load(embeddings_pickle)
		vectorizer = features.get_embed(embeddings, pool = 'pool')
	elif feature == "union":
		embeddings_pickle = open("vectors-320.pickle","rb")
		embeddings = pickle.load(embeddings_pickle)
		tf_idf = FeatureUnion([('tfidf_w',TfidfVectorizer(max_features=2500, ngram_range = (1,3))),('tfidf_c',TfidfVectorizer(max_features=2500, ngram_range = (2,5), analyzer = 'char'))])
		vectorizer = FeatureUnion([('tfidf',tf_idf),('embedding',features.get_embed(embeddings, pool = 'pool'))])

	if feature == "embeddings":
		clf = LinearSVC(C=1)
	else:
		clf = LinearSVC(C=0.1)
	classifier = Pipeline([('vectorize', vectorizer),('classify', clf)])
	#return(cross_val_score(classifier, X_train,Y_train, cv=10, scoring='accuracy'))
	print('Predicting...')
	#return(classifier.predict(X_test))
	return(cross_val_score(classifier,X_train,Y_train,cv=10))


def train_and_predict(X_train,Y_train,X_test,Y_test,feature):
	print("Training...")
	if feature == "tfidf":
		#vectorizer = TfidfVectorizer(max_features=2500, ngram_range = (1,3))#, stop_words = stop_set)
		vectorizer = FeatureUnion([('tfidf_w',TfidfVectorizer(max_features=2500, ngram_range = (1,3))),('tfidf_c',TfidfVectorizer(max_features=2500, ngram_range = (2,5), analyzer = 'char'))])
	elif feature == "embeddings":
		embeddings_pickle = open("vectors-320.pickle","rb")
		embeddings = pickle.load(embeddings_pickle)
		vectorizer = features.get_embed(embeddings, pool = 'pool')
	elif feature == "union":
		embeddings_pickle = open("vectors-320.pickle","rb")
		embeddings = pickle.load(embeddings_pickle)
		tf_idf = FeatureUnion([('tfidf_w',TfidfVectorizer(max_features=2500, ngram_range = (1,3))),('tfidf_c',TfidfVectorizer(max_features=2500, ngram_range = (2,5), analyzer = 'char'))])
		vectorizer = FeatureUnion([('tfidf',tf_idf),('embedding',features.get_embed(embeddings, pool = 'pool'))])

	clf = LinearSVC(C=0.1)
	classifier = Pipeline([('vectorize', vectorizer),('classify', clf)])
	print("Predicting...")
	classifier.fit(X_train,Y_train)
	return(classifier.predict(X_test))


def main():
	score = 0
	#try:
	X_train, X_test, Y_train, Y_test = open_corpus("tweets.pickle")
	develop_score = runSVC(X_train,Y_train, sys.argv[1])
	print(mean(develop_score))
	#Y_guess = train_and_predict(X_train,Y_train,X_test,Y_test,sys.argv[1])
	#print(evaluate(Y_guess, Y_test))

	#except UnboundLocalError:
	#	print("Please use the right format and give the type of vectorizer as argument.")

if __name__ == '__main__':
	main()
