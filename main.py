#!/usr/bin/env python

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import cross_val_score,cross_val_predict
from statistics import mean
import random
import features
#from sklearn.base import TransformerMixin
import pickle
import sys

def open_corpus(pickle_path):
	print("\n##### Reading corpus...")
	print("Reading file:", pickle_path)
	x = list()
	y = list()
	with open(pickle_path, 'rb') as annotated_pickle:
		for tweet,annotation in pickle.load(annotated_pickle).items():
			x.append(tweet)
			y.append(annotation)
	c = list(zip(x, y))
	c = sorted(c)
	random.seed(85)
	random.shuffle(c)
	x,y = zip(*c)
	X_train, Y_train = x[:800], y[:800]
	X_test, Y_test = x[800:], y[800:]
	return X_train, X_test, Y_train, Y_test


def runSVC(X_train,Y_train, feature):
	stop_set = set(stopwords.words("dutch"))
	if feature == "tfidf":
		#vectorizer = TfidfVectorizer(max_features=2500, ngram_range = (1,5))
		vectorizer = FeatureUnion([('tfidf_w',TfidfVectorizer(max_features=2500, ngram_range = (1,3))),('tfidf_c',TfidfVectorizer(max_features=2500, ngram_range = (2,5), analyzer = 'char'))])#, stop_words = stop_set)
	elif feature == "embeddings":
		embeddings_pickle = open("vectors-320-2.pickle","rb")
		embeddings = pickle.load(embeddings_pickle)
		vectorizer = features.get_embed(embeddings,pool = 'pool')
	elif feature == "union":
		embeddings_pickle = open("vectors-320-2.pickle","rb")
		embeddings = pickle.load(embeddings_pickle)
		vectorizer = FeatureUnion([('tfidf',TfidfVectorizer(max_features=1000,use_idf = False)),('embedding',features.get_embed(embeddings, pool = 'pool'))])


	clf = LinearSVC()
	classifier = Pipeline([('vectorize', vectorizer),('classify', clf)])
	#return(cross_val_score(classifier, X_train,Y_train, cv=10, scoring='accuracy'))
	print('Predicting...')
	#return(classifier.predict(X_test))
	return(cross_val_score(classifier,X_train,Y_train,cv=10))


def train_and_predict(X_train,Y_train,X_test,Y_test,feature):
	if feature == "tfidf":
		vectorizer = TfidfVectorizer(max_features=2500, use_idf=False)
	elif feature == "embeddings":
		embeddings_pickle = open("vectors-320-2.pickle","rb")
		embeddings = pickle.load(embeddings_pickle)
		vectorizer = features.get_embed(embeddings)
	elif feature == "union":
		embeddings_pickle = open("vectors-320-2.pickle","rb")
		embeddings = pickle.load(embeddings_pickle)
		vectorizer = FeatureUnion([('tfidf',TfidfVectorizer(max_features=1000, use_idf=False)),('embedding',features.get_embed(embeddings, pool = 'pool'))])

	clf = LinearSVC()
	classifier = Pipeline([('vectorize', vectorizer),('classify', clf)])
	classifier.fit(X_train,Y_train)
	print("Predicting...")
	classifier.predict(X_test)


def main():
	score = 0
	#try:
	X_train, X_test, Y_train, Y_test = open_corpus("tweets.pickle")
	Y_guess = runSVC(X_train,Y_train, sys.argv[1])
	print(mean(Y_guess))
	#for i in range(len(Y_guess)):
	#	if Y_guess[i] == Y_test[i]:
	#		score+=1
	#print(score)
	#except UnboundLocalError:
	#	print(sys.argv[1])
	#	print("Please use the right format and give the type of vectorizer as argument.")

if __name__ == '__main__':
	main()
