import pandas as pd
import re
import numpy as np
import string
import pickle
import Data_Cleaning
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support, roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score


def sample_dataframe():
    sample = ["fuck fuck fuck", "cunt", "happy place", "this is awesome i love this so much!!",
              "what the fuck are you doing you asshole", "wow this is so beautiful",
              "aa another soar loser who lives in the cry baby world",
              "fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck",
              "What a stupid article. I this what passes as scholarship?",
              "Oh, puleeze - are we to understand the only qualification for office is the candidate's gender?",
              "Sorry. I meant a water pipeline from Canada to California.",
              "you are a stupid fuck and your mother's cunt stink"]

    return pd.Series(sample)


def test_data():
    pass


def main():
    comments = pd.read_csv("mergedDataSet.csv", encoding='ISO-8859-1')#[:100]
    fb_data = pd.read_csv("Arsenal_facebook_comments.csv", encoding='ISO-8859-1')[["comment_id", "comment_message"]]
    X_test_cleaned = fb_data.copy()
    X_test_cleaned["comment_message"] = X_test_cleaned["comment_message"].apply(lambda x: Data_Cleaning.text_cleaning(x))

    X_test_fb = X_test_cleaned[X_test_cleaned["comment_message"].apply(lambda x: x != "")]

    X_train, X_test, y_train, y_test = train_test_split(comments["comment_text"], comments["merged_rating"], random_state=0)
    stop_words = open("stop_words.txt", "r").read().split()

    v = TfidfVectorizer(use_idf=True, max_df=0.7, lowercase=True, stop_words=stop_words, strip_accents="unicode",
                        token_pattern=r"(?u)\b\w*[a-zA-Z]\w*\b", ngram_range=(1, 2))

    v.fit(X_train)

    x_train_tfidf = v.transform(X_train)

    sm = SMOTE(random_state=42)
    x_data, y_data = sm.fit_sample(x_train_tfidf, y_train)

    clf = RandomForestClassifier(random_state=0, n_jobs=-1).fit(x_data, y_data)

    pkl_filename = "pickle_randomforest_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(clf, file)

    # pkl_filename = "pickle_model.pkl"
    # with open(pkl_filename, 'wb') as file:
    #     pickle.dump(v, file)

    x_test_tfidf = v.transform(X_test)
    # print(x_test_tfidf)
    # #
    y_predict = clf.predict(x_test_tfidf)
    print("WEIGHTED", precision_recall_fscore_support(y_test, y_predict, average="weighted"))
    print("MACRO", precision_recall_fscore_support(y_test, y_predict, average="macro"))

    # roc_curve(y_test, y_predict)
    # x_fb_tfidf = v.transform(X_test_fb["comment_message"])
    # y_fb_predict = clf.predict(x_fb_tfidf)
    # fb_test_df = pd.DataFrame(X_test_fb)

    # fb_test_df["predictions"] = y_fb_predict
    # merged_fb_predictions = pd.DataFrame.merge(fb_data, fb_test_df, on="comment_id", how="inner")
    # print(merged_fb_predictions)

    #merged_fb_predictions.to_csv("facebook_predictions_noreg.csv")

    #print(y_fb_predict)


if __name__ == "__main__":
    main()