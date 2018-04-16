import pickle
import pandas as pd
import Data_Cleaning
from sklearn.metrics import precision_recall_fscore_support, roc_curve
from sklearn.model_selection import train_test_split


def sample_dataframe():
    sample = ["fuck fuck fuck", "cunt", "happy place", "this is awesome i love this so much!!",
              "what the fuck are you doing you asshole", "wow this is so beautiful",
              "aa another soar loser who lives in the cry baby world",
              "fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck",
              "What a stupid article. I this what passes as scholarship?",
              "Oh, puleeze - are we to understand the only qualification for office is the candidate's gender?",
              "Sorry. I meant a water pipeline from Canada to California.",
              "you are a stupid fuck and your mother's cunt stink",
              "fork you"]

    return pd.Series(sample)


def main():
    comments = pd.read_csv("mergedDataSet.csv", encoding='ISO-8859-1')#[:100]
    X_train, X_test, y_train, y_test = train_test_split(comments["comment_text"], comments["merged_rating"], random_state=0)

    pkl_filename = "pickle_tfidf_model.pkl"
    with open(pkl_filename, 'rb') as file:
        pickle_tfidf_model = pickle.load(file)

    pkl_filename = "pickle_randomforest_model.pkl"
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)

    x_test_tfidf = pickle_tfidf_model.transform(X_test)

    y_predict = pickle_model.predict(x_test_tfidf)
    # print("WEIGHTED", precision_recall_fscore_support(y_test, y_predict, average="weighted"))
    # print("MACRO", precision_recall_fscore_support(y_test, y_predict, average="macro"))

    # x_test_tfidf = pickle_tfidf_model.transform(sample_dataframe())
    # y_predict = pickle_model.predict(x_test_tfidf)
    # print(y_predict)
    #
    # fb_data = pd.read_csv("Arsenal_facebook_comments.csv", encoding='ISO-8859-1')[["comment_id", "comment_message"]]
    # X_test_cleaned = fb_data.copy()
    # X_test_cleaned["comment_message"] = X_test_cleaned["comment_message"].apply(lambda x: Data_Cleaning.text_cleaning(x))
    #
    # X_test_fb = X_test_cleaned[X_test_cleaned["comment_message"].apply(lambda x: x != "")]

    print(roc_curve(y_test, y_predict))


    # x_fb_tfidf = pickle_tfidf_model.transform(X_test_fb["comment_message"])
    # y_fb_predict = pickle_model.predict(x_fb_tfidf)
    # fb_test_df = pd.DataFrame(X_test_fb)
    #
    # fb_test_df["predictions"] = y_fb_predict
    # merged_fb_predictions = pd.DataFrame.merge(fb_data, fb_test_df, on="comment_id", how="inner")
    # print(merged_fb_predictions)
    #
    # merged_fb_predictions.to_csv("facebook_predictions_randomforest.csv")


if __name__ == "__main__":
    main()