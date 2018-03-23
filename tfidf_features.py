import pandas as pd
import re
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn import metrics


def top_tfidf_feats(row, features, top_n):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [features[i] for i in topn_ids]
    return top_feats


def top_feats_in_doc(Xtr, features, row_id, top_n):
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)


def main():
    comments = pd.read_csv("mergedDataSet.csv", encoding='ISO-8859-1')
    #X_train, X_test, y_train, y_test = train_test_split(comments["comment_text"], comments["merged_rating"], random_state=0)

    v = TfidfVectorizer(use_idf=True, max_df=0.7, lowercase=True, stop_words="english", strip_accents="unicode",
                        token_pattern=r"(?u)\b\w*[a-zA-Z]\w*\b", ngram_range=(1, 2))
    #v.fit(X_train)
    v.fit(comments["comment_text"])
    #x_train_tfidf = v.transform(X_train)
    x_tfidf = v.transform(comments["comment_text"])

    #clf = MultinomialNB().fit(x_train_tfidf, y_train)
    y = comments["merged_rating"]
    clf = MultinomialNB() #.fit(x_tfidf, comments["merged_rating"])
    #x_test_tfidf = v.transform(X_test)

    #y_predict = clf.predict(x_test_tfidf)
    #print(metrics.accuracy_score(y_test, y_predict))

    scores = cross_val_score(clf, x_tfidf, y, cv=10, scoring='accuracy')
    print(scores.mean())
    # feature_array = x.toarray()
    # np.set_printoptions(threshold=np.nan)
    #print(np.asarray(v.get_feature_names()))

    # feature_names = v.get_feature_names()
    # top_words = top_feats_in_doc(x, feature_names, 2, top_n=20)
    # print(x[1].vocabulary_)


if __name__ == "__main__":
    main()