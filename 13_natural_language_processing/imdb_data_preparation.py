import pandas as pd
import numpy as np
import re
import string
import nltk
import nltk.corpus as nltk_corpus


imdb_reviews = pd.read_csv("data/IMDB Dataset.csv")

# imdb_review_count = imdb_reviews.groupby(by="sentiment").count()
# print(imdb_review_count)

# N = 1000
N = len(imdb_reviews)
X = imdb_reviews["review"].iloc[:N]
y = imdb_reviews["sentiment"].iloc[:N]

y.replace({"positive": 1, "negative": 0}, inplace=True)
X = np.array(X)

# stemmer = nltk.stem.LancasterStemmer()
# stemmer = nltk.stem.PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()

stop_word = nltk_corpus.stopwords.words("english")
# print(stop_word)

for x_row in range(len(X)):
    X[x_row] = re.sub("<.*?>", " ", X[x_row])
    X[x_row] = X[x_row].lower()
    X[x_row] = X[x_row].translate(str.maketrans("", "", string.punctuation))
    
    X[x_row] = nltk.word_tokenize(X[x_row])
    
    # X[x_row] = [stemmer.stem(word) for word in X[x_row]]
    X[x_row] = [lemmatizer.lemmatize(word) for word in X[x_row]]
    
    X[x_row] = " ".join(X[x_row])

    if (x_row + 1) % 100 == 0:
        print("{}/{} reviews prepared.".format(x_row+1, len(X)))
else:
    print("Preparation finished.")

imdb_reviews["review"] = X
imdb_reviews["sentiment"] = y

imdb_reviews.to_csv("data/imdb_dataset_prepared.csv", index=False)
