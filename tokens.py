#!/usr/bin/env python

import string
import pickle

def main():
	with open("tokens.txt","w") as f:
		with open("annotated.txt","r",encoding='latin-1') as c:
			tokens = set()
			for line in c:
				tweet = line.split("\t")[1]
				for token in tweet.split():
					if token.isalpha():
						tokens.add(token)
					else:
						new_token= ""
						for char in token:
							if char in string.punctuation:
								pass
							else:
								new_token+=char
						if new_token.isalpha() and new_token!= "":
							tokens.add(new_token)
		for token in tokens:
			f.write(token+"\n")

if __name__ == '__main__':
    main()
