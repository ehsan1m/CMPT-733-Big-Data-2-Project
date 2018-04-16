import pandas as pd
import numpy as np
import operator
import re
import csv
import Data_Cleaning
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#
# df_merge = pd.read_csv("mergedDataSet.csv")
# df_toxic = df_merge[(df_merge['merged_rating'] == 1) | (df_merge['merged_rating'] == 2)][
#     ['comment_text', 'merged_rating']]


def get_toxicity_types(df, df_slur):
    df = df[df["prediction"] != 0]
    df["comment_message_cleaned"] = df["comment_message"].apply(lambda x: Data_Cleaning.text_cleaning(x))
    df = df[df["comment_message_cleaned"].apply(lambda x: x is not "")]

    df["homophobic"] = np.zeros(len(df))
    df["sexist"] = np.zeros(len(df))
    df["racist"] = np.zeros(len(df))

    df_slur = df_slur.fillna('None')
    homophobic_array = df_slur['Homophobic'].str.lower().values
    sexist_array = df_slur['Sexist'].str.lower().values
    racist_array = df_slur['Racist'].str.lower().values

    comments_array = df["comment_message_cleaned"].values

    for idx, word in enumerate(comments_array):
        for w in word.split():
            if w in racist_array:
                df["racist"].iloc[idx] = 1
            if w in sexist_array:
                df["sexist"].iloc[idx] = 1
            if w in homophobic_array:
                df["homophobic"].iloc[idx] = 1

    return df


def get_value(x):
    if True in x.index:
        return x.get_value(True)
    return 0


def get_statistics(df, total_size):
    toxicity_types = {}
    df_len = len(df)

    toxicity_types["racist"] = (df["racist"].sum() / df_len) * 100
    toxicity_types["sexist"] = (df["sexist"].sum() / df_len) * 100
    toxicity_types["homophobic"] = (df["homophobic"].sum() / df_len) * 100

    # toxicity_types["racist_and_sexist"] = get_value(((df["racist"] == 1) & (df["sexist"] == 1)).value_counts())
    # toxicity_types["racist_and_homophobic"] = get_value(((df["racist"] == 1) & (df["homophobic"] == 1)).value_counts())
    # toxicity_types["sexist_and_homophobic"] = get_value(((df["sexist"] == 1) & (df["homophobic"] == 1)).value_counts())
    #
    # toxicity_types["racist_and_sexist"] = (toxicity_types["racist_and_sexist"] / df_len) * 100
    # toxicity_types["racist_and_homophobic"] = (toxicity_types["racist_and_homophobic"] / df_len) * 100
    # toxicity_types["sexist_and_homophobic"] = (toxicity_types["sexist_and_homophobic"] / df_len) * 100
    return toxicity_types


def output_confidence_interval(sports_stats, news_stats, entertainment_stats):
    stats_df = pd.DataFrame([sports_stats, news_stats, entertainment_stats])
    stats_df.to_csv("statistics_confidence.csv", index=None)


def output_bar_chart(sports_stats, news_stats, entertainment_stats, sports_total, news_total, enter_total,
                     sports_toxic, news_toxic, enter_toxic):
    # output for raman
    with open("statistics.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "Type", "Percentage", "Toxicity", "Total"])
        for key, value in sports_stats.items():
            writer.writerow(["Sports", key, value, sports_toxic, sports_total])
        for key, value in news_stats.items():
            writer.writerow(["News", key, value, news_toxic, news_total])
        for key, value in entertainment_stats.items():
            writer.writerow(["Entertainment", key, value, enter_toxic, enter_total])


def main():
    df_slur = pd.read_csv("Slangs List Big Data.csv", encoding='ISO-8859-1')
    sports_comments = pd.DataFrame.from_csv("Sports_Predictions.csv")#[:100]
    news_comments = pd.DataFrame.from_csv("News_Predictions.csv")#[:100]
    entertainment_comments = pd.DataFrame.from_csv("Entertainment_Predictions.csv")#[:100]

    sports_df = get_toxicity_types(sports_comments, df_slur)
    news_df = get_toxicity_types(news_comments, df_slur)
    entertainment_df = get_toxicity_types(entertainment_comments, df_slur)

    sports_stats = get_statistics(sports_df, len(sports_comments))
    news_stats = get_statistics(news_df, len(news_comments))
    entertainment_stats = get_statistics(entertainment_df, len(entertainment_comments))

    output_bar_chart(sports_stats, news_stats, entertainment_stats, len(sports_comments), len(news_comments),
                     len(entertainment_comments), len(sports_df), len(news_df), len(entertainment_df))

    # sports_stats["Category"] = "Sports"
    # news_stats["Category"] = "News"
    # entertainment_stats["Category"] = "Entertainment"
    #
    # output_confidence_interval(sports_stats, news_stats, entertainment_stats)

    print(sports_stats)
    print(news_stats)
    print(entertainment_stats)


if __name__ == "__main__":
    main()
