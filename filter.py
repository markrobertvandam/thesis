#!/usr/bin/python3

z = 0
with open("tweets.txt") as f:
	with open("filtered_tweets.txt","w") as c:
		tweets = f.read().split("|")
		for i in tweets:
			i = i.lower()
			if "#stempvv" not in i and "\n" not in i and "#wegmetpvv" not in i and "#pvvnee" not in i and "#pvvnoway" not in i and "#pvvisnsb" not in i and "#pvvisgeenpartij" not in i and "http" not in i and "#pvvnot" not in i and ".nl" not in i and "turk" not in i and len(i) > 50 and "maurice" not in i and "dwdd" not in i and not "zelfmoord" in i and "jinek" not in i and "vvd" not in i and not "stembusfraude" in i and not "stabiel midden" in i and not "funny" in i:
				c.write(i+"\n")
				z+=1
				if z == 1001:
					break	
