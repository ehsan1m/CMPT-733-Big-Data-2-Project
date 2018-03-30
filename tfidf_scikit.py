import pandas as pd
import re
import numpy as np
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support
from imblearn.over_sampling import SMOTE


def sample_dataframe():
    sample = ["fuck fuck fuck", "cunt", "happy place", "this is awesome i love this so much!!",
              "what the fuck are you doing you asshole", "wow this is so beautiful",
              "fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck"]

    return pd.Series(sample)


def main():
    comments = pd.read_csv("mergedDataSet.csv", encoding='ISO-8859-1')#[:100]
    X_train, X_test, y_train, y_test = train_test_split(comments["comment_text"], comments["merged_rating"], random_state=0)
    stop_words = open("stop_words.txt", "r").read().split()

    v = TfidfVectorizer(use_idf=True, max_df=0.7, lowercase=True, stop_words=stop_words, strip_accents="unicode",
                        token_pattern=r"(?u)\b\w*[a-zA-Z]\w*\b", ngram_range=(1, 2))

    v.fit(X_train)

    x_train_tfidf = v.transform(X_train)

    sm = SMOTE(random_state=42)
    x_data, y_data = sm.fit_sample(x_train_tfidf, y_train)

    clf = MultinomialNB().fit(x_data, y_data)

    # pkl_filename = "pickle_model.pkl"
    # with open(pkl_filename, 'wb') as file:
    #     pickle.dump(v, file)
    #
    # pkl_filename = "pickle_tfidf_model.pkl"
    # with open(pkl_filename, 'rb') as file:
    #     pickle_model = pickle.load(file)

    x_test_tfidf = v.transform(X_test)
    # print(x_test_tfidf)
    # #
    y_predict = clf.predict(x_test_tfidf)
    print("WEIGHTED", precision_recall_fscore_support(y_test, y_predict, average="weighted"))
    print("MACRO", precision_recall_fscore_support(y_test, y_predict, average="macro"))


    x_test_tfidf = v.transform(sample_dataframe())
    y_predict = clf.predict(x_test_tfidf)
    print(y_predict)
    #print("WEIGHTED", precision_recall_fscore_support(y_test, y_predict, average="weighted"))
    #print("MACRO", precision_recall_fscore_support(y_test, y_predict, average="macro"))


if __name__ == "__main__":
    main()