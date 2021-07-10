import pandas as pd
import numpy as np
import re
import string
import nltk
import nltk.corpus as nltk_corpus


imdb_reviews = pd.read_csv("data/IMDB Dataset.csv")

X = imdb_reviews["review"].iloc[3]
print(X, "\n")

X = re.sub("<.*?>", " ", X)
X = X.lower()
X = X.translate(str.maketrans("", "", string.punctuation))

# stemmer = nltk.stem.LancasterStemmer()
stemmer = nltk.stem.PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()

stop_word = nltk_corpus.stopwords.words("english")
# print(stop_word)

X = nltk.word_tokenize(X)

# X = [stemmer.stem(word) for word in X]
X = [lemmatizer.lemmatize(word) for word in X]

X = " ".join(X)

print(X, "\n")
