#!/usr/bin/env python

from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import cross_val_score,cross_val_predict
from statistics import mean
from sklearn.model_selection import GridSearchCV
from prettytable import PrettyTable
import numpy as np
import random
import features
import pickle
import sys

def open_corpus(pickle_path):
	"""
	Loads the dataset and creates train and test data.
	"""
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

def calculate_f1(precision,recall):
	"""
	Function to calculate the f1-scores.
	""" 
	return(round((2 * precision * recall) / (precision + recall),3))

def evaluate(Y_guess, Y_test):
	"""
	Evaluation function
	""" 
	cm = np.zeros((3, 3), dtype=int)
	np.add.at(cm, [Y_test, Y_guess], 1)
	precision_neutral = cm[0,0] / (cm[0,0] + cm[1,0]+cm[2,0])
	recall_neutral = cm[0,0] / (cm[0,0] + cm[0,1] + cm[0,2])
	precision_favor = cm[1,1] / (cm[1,1] + cm[0,1]+cm[2,1])
	recall_favor = cm[1,1] / (cm[1,1] + cm[1,0] + cm[1,2])
	precision_against = cm[2,2] / (cm[2,2] + cm[0,2]+cm[1,2])
	recall_against = cm[2,2] / (cm[2,2] + cm[2,0] + cm[2,1])
	p_r_table = PrettyTable(['Class','Precision','Recall', 'F_1'])
	p_r_table.add_row(['Neutral',round(precision_neutral,3),round(recall_neutral,3), calculate_f1(precision_neutral,recall_neutral)])
	p_r_table.add_row(['Favor',round(precision_favor,3),round(recall_favor,3), calculate_f1(precision_favor,recall_favor)])
	p_r_table.add_row(['Against',round(precision_against,3),round(recall_against,3), calculate_f1(precision_against,recall_against)])
	print(p_r_table, "\n")
	print("Overall accuracy: ", (cm[0,0] + cm[1,1] + cm[2,2]) / 200)
	print("Average F1_score: ", round((calculate_f1(precision_favor,recall_favor) + calculate_f1(precision_against,recall_against)) / 2,3))
	

def runSVC(X_train,Y_train, feature):
	"""
	Runs the LinearSVC for development with a cross-fold validation.
	""" 
	stop_set = set(stopwords.words("dutch"))
	if feature == "tfidf":
		#vectorizer = TfidfVectorizer(max_features=2500, ngram_range = (1,3))#, stop_words = stop_set)
		vectorizer = FeatureUnion([('tfidf_w',TfidfVectorizer(max_features=2500, ngram_range = (1,3), stop_words = stop_set)),('tfidf_c',TfidfVectorizer(max_features=2500, ngram_range = (2,5), analyzer = 'char'))])
	elif feature == "embeddings":
		embeddings_pickle = open("vectors-320.pickle","rb")
		embeddings = pickle.load(embeddings_pickle)
		vectorizer = features.get_embed(embeddings, pool = 'pool')
	elif feature == "union":
		embeddings_pickle = open("vectors-320.pickle","rb")
		embeddings = pickle.load(embeddings_pickle)
		tf_idf = FeatureUnion([('tfidf_w',TfidfVectorizer(max_features=2500, ngram_range = (1,3), stop_words = stop_set)),('tfidf_c',TfidfVectorizer(max_features=2500, ngram_range = (2,5), analyzer = 'char'))])
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
	"""
	The actual training and prediction of the final classifier.
	""" 
	print("Training...")
	stop_set = set(stopwords.words("dutch"))
	if feature == "tfidf":
		#vectorizer = TfidfVectorizer(max_features=2500, ngram_range = (1,3))
		vectorizer = FeatureUnion([('tfidf_w',TfidfVectorizer(max_features=2500, ngram_range = (1,3), stop_words = stop_set)),('tfidf_c',TfidfVectorizer(max_features=2500, ngram_range = (4,6), analyzer = 'char'))])
	elif feature == "embeddings":
		embeddings_pickle = open("vectors-320.pickle","rb")
		embeddings = pickle.load(embeddings_pickle)
		vectorizer = features.get_embed(embeddings, pool = 'pool')
	elif feature == "union":
		embeddings_pickle = open("vectors-320.pickle","rb")
		embeddings = pickle.load(embeddings_pickle)
		tf_idf = FeatureUnion([('tfidf_w',TfidfVectorizer(max_features=2500, ngram_range = (1,3), stop_words = stop_set)),('tfidf_c',TfidfVectorizer(max_features=2500, ngram_range = (4,6), analyzer = 'char'))])
		vectorizer = FeatureUnion([('tfidf',tf_idf),('embedding',features.get_embed(embeddings, pool = 'pool'))])

	if feature == "embeddings":
		clf = LinearSVC(C=1)
	else:
		clf = LinearSVC(C=0.1)
	classifier = Pipeline([('vectorize', vectorizer),('classify', clf)])
	print("Predicting...")
	classifier.fit(X_train,Y_train)
	return(classifier.predict(X_test))


def main():
	score = 0
	#try:
	X_train, X_test, Y_train, Y_test = open_corpus("tweets.pickle")
	#develop_score = runSVC(X_train,Y_train, sys.argv[1])
	#print(mean(develop_score))
	Y_guess = train_and_predict(X_train,Y_train,X_test,Y_test,sys.argv[1])
	evaluate(Y_guess, Y_test)

	#except UnboundLocalError:
	#	print("Please use the right format and give the type of vectorizer as argument.")

if __name__ == '__main__':
	main()
