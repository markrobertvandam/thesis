#!/usr/bin/env python

import pickle

def main():
	annotated_dict = {}
	with open ("annotated.txt","r") as f:
		for line in f:
			item = line.rstrip().split("\t")
			if item[2].endswith("-1"):
				annotation = -1
			else:
				annotation = int(item[2][-1])
			annotated_dict[item[1]] = annotation
	print(annotated_dict)
	tweets_pickle = open("tweets.pickle","wb")
	pickle.dump(annotated_dict,tweets_pickle)
	tweets_pickle.close()	

if __name__ == '__main__':
    main()
