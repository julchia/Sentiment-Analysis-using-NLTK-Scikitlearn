import pandas as pd
import re
import random

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer

from nltk import classify
from nltk import NaiveBayesClassifier

cols = ['sentiment','id','date','query_string','user','text']
data = pd.read_csv('0_TWEET_COMMENTS.csv', names=cols, encoding='ISO-8859-1')

pos_tweets = data[ data['sentiment'] == 4]
pos_tweets = pos_tweets['text']
neg_tweets = data[ data['sentiment'] == 0]
neg_tweets = neg_tweets['text']

def clean_tweets(tweetlist):

    clean_tweets = []

    # Tokenize words.
    tweet_tok = [nltk.word_tokenize(str(sentence)) for sentence in tweetlist]

    # Lemmatize tokens (find lemma of each token), remove stopwords and reduce everything to lowercase.
    bag_of_words = [word for sentece in tweet_tok for word in sentece]

    for token, tag in nltk.pos_tag(bag_of_words):

        token = re.sub(r'[^(a-zA-Z)\s]','', token)

        if tag.startswith('NN'):    
                pos = 'n'
        elif tag.startswith('VB'):
                pos = 'v'
        else:
                pos = 'a'
        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        stopwords_set = set(stopwords.words('english'))

        if len(token) > 2 and 'http' not in token and token != 'RT' and not token in stopwords_set:
            clean_tweets.append(token.lower())
        
    return clean_tweets

# Filter words depending on their type, in this case, I select adjectives.
def filter_tweets(cleaned_tweetlist):
    tweets_tag = nltk.pos_tag(cleaned_tweetlist)
    allowed_words = ['JJ']
    allowed_tweets = []
    for w in tweets_tag:
        if w[1] in allowed_words:
            allowed_tweets.append(word_tokenize(w[0]))
    return allowed_tweets

# ---------------------------------- Prepossessing ---------------------------------- <> <> <> <> <>

pos_sample = pos_tweets[:10000]
neg_sample = neg_tweets[:10000]

# Clean and filter tweets
pos_final = filter_tweets(clean_tweets(pos_sample))
neg_final = filter_tweets(clean_tweets(neg_sample))

# ---------------------------------- Train and test model ---------------------------------- <> <> <> <> <>

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

positive_tokens_for_model_1 = get_tweets_for_model(pos_final)
negative_tokens_for_model_1 = get_tweets_for_model(neg_final)

positive_dataset = [(tweet_dict, 'pos') for tweet_dict in positive_tokens_for_model_1]
negative_dataset = [(tweet_dict, 'neg') for tweet_dict in negative_tokens_for_model_1]

dataset = positive_dataset + negative_dataset
random.shuffle(dataset)

train_data = dataset[:5000]
test_data = dataset[5000:]

classifier = NaiveBayesClassifier.train(train_data)
print("Accuracy is:", classify.accuracy(classifier, test_data))
print(classifier.show_most_informative_features(10))