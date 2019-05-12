#!/usr/bin/env python

import pickle

def main():
	annotated_dict = {}
	cnt = 0
	with open ("annotated.txt","r") as f:
		for line in f:
			item = line.rstrip().split("\t")
			if item[2].endswith("-1"):
				annotation = -1
			else:
				annotation = int(item[2][-1])
			if item[1] in annotated_dict:
				print(item[1])
			annotated_dict[item[1]] = annotation
	tweets_pickle = open("tweets.pickle","wb")
	pickle.dump(annotated_dict,tweets_pickle)
	tweets_pickle.close()	

if __name__ == '__main__':
    main()
