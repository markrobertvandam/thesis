#!/usr/bin/env python

class get_embed():
	'''
	class used to create the word embeddings vectorizer
	'''
	def __init__(self, word_embeds, pool):
		self.word_embeds = word_embeds
		self.pool_method = pool
	
	def transform(self,X, **transform_params):
		return([self.get_embedding(sentence,self.word_embeds, self.pool_method) for sentence in X])

	def fit(self, X, y=None, **fit_params):
		return self

	def get_embedding(self, sentence, word_embeds, pool):
		embedded_sent = [word_embeds[word] for word in sentence.split() if word in word_embeds]
		if pool == 'pool':
			sent_embedding = [sum(col) / float(len(col)) for col in zip(*embedded_sent)]  # average pooling
		elif pool == 'max':
			sent_embedding = [max(col) for col in zip(*embedded_sent)]	# max pooling
		else:
			raise ValueError('Unknown pooling method!')

		return sent_embedding



