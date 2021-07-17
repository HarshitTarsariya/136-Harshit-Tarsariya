import nltk, scipy, pandas, numpy
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt

positive=twitter_samples.strings('positive_tweets.json')
negative=twitter_samples.strings('negative_tweets.json')

print(f"Number of + Tweets:{len(positive)}")
print(f"Number of - Tweets:{len(negative)}")

# fig=plt.figure(figsize=(5,5))
# labels='A','B','C','D'
# sizes=['33','33','33','1']

# plt.pie(sizes,labels=labels,shadow=True,startangle=90)
# plt.axis('equal')
# plt.show()

labels='Positive','Negative'
size=[len(positive),len(negative)]
plt.pie(size, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
plt.axis('equal')

# plt.show()

# print('\033[92m'+positive[0])
# print('\033[91m'+negative[0])

nltk.download('stopwords')

import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

tweet=positive[2277]
print(f'\nOriginal :{tweet}')

tweet2=re.sub(r'https?:\/\/.*[\r\n]*','',tweet)
print(f'\nURLs removed :{tweet2}')

tweet2=re.sub(r'#','',tweet2)
print(f'\nHashtags removed :{tweet2}')


#Tokenizer

tokenizer=TweetTokenizer(preserve_case=False)

tweet_tokens=tokenizer.tokenize(tweet2)
print(f'\n Tokens:{tweet_tokens}')

stopwords_english=stopwords.words('english')

print(f'\nEnglish StopWords:{stopwords_english}')

tweet_clean=[]

for word in tweet_tokens:
    if(word not in stopwords_english and word not in string.punctuation):
        tweet_clean.append(word)

print(f'\nRemoved StopWords and Punctions:{tweet_clean}')

#Stemming: Bringing to its root form 
#learning,learned,learnt ---> learn

stemmer=PorterStemmer()

tweet_stem=[]

for word in tweet_clean:
    stem_word=stemmer.stem(word)
    tweet_stem.append(stem_word)
print(f'\nStemmed Words:{tweet_stem}')






