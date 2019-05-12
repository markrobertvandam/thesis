#!/usr/bin/env python

import string
import pickle
import spacy

def main():
	with open("tokens.txt","w") as f:
		with open("annotated.txt","r",encoding='latin-1') as c:
			tokens = set()
			nlp = spacy.load('nl_core_news_sm')
			for line in c:
				tweet = line.split("\t")[1]
				for token in nlp(tweet):
					if token.text.isalpha():
						tokens.add(token.text)
					else:
						new_token= ""
						for char in token.text:
							if char in string.punctuation:
								pass
							else:
								new_token+=char
						if new_token.isalpha() and new_token!= "":
							tokens.add(new_token)
		for token in tokens:
			f.write(token+"\n")

#	with open("wikipedia-160.txt","r") as f:
#		for line in f:
#			print(line.split()[0])



if __name__ == '__main__':
    main()
