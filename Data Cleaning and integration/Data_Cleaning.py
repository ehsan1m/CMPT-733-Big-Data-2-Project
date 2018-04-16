import pandas as pd
import re
from nltk.stem import WordNetLemmatizer

stop_words = open("stop_words.txt", "r").read().split()
lemmatizer = WordNetLemmatizer()


# Function to read comments file of different pages
def read_file(file_path):
    df = pd.read_csv(file_path)
    return df


# Function to Clean Data

def text_cleaning(comment):
    comment = comment.lower()  # Converting comments into lowercase
    comment = comment.strip("b'").strip('b"')  # Removing b",b' from start and end of comment
    comment = re.sub("\\[\\[(.*?)\\]\\]", "", comment)  # Removing GIFs and images from comments
    comment = re.sub(r'\\x\S+', "", comment)  # Removing unwanted text , emojis etc.
    #   Replacing apostrophes
    comment = re.sub(r"'s", ' is', comment)
    comment = re.sub(r"'re", ' are', comment)
    comment = re.sub(r"'t", ' not', comment)
    comment = re.sub(r"'m", ' am', comment)
    comment = re.sub(r"'d", ' would', comment)
    comment = re.sub(r"'ll", ' will', comment)
    comment = re.sub(r"'ve", ' have', comment)
    comment = re.sub('[.]', ' ', comment)

    comment = ''.join([c for c in comment if c not in (
    '!', '?', '.', '\\', '"', ',', '$', '%', "'")])  # Removing Punctuations and other signs
    comment = re.sub(r'[0-9]', "", comment)  # Removing numbers
    comment = re.sub(r'http\S+', "", comment)  # Removing Url
    comment = ' '.join([c for c in comment.split() if c not in stop_words and len(c) > 2])  # Removing stopwords
    comment = ' '.join([lemmatizer.lemmatize(c) for c in comment.split()])  # Lemmatizing
    return comment


def sample_df(page_list):
    chunks = []
    for page in page_list:
        df_page = read_file(page)
        chunks.append(df_page)
    df_concat = pd.concat(chunks, ignore_index=True)
    df_concat = df_concat['comment_message']#.sample(n=20000, random_state=10)
    return df_concat


def merge_fb_data():
    # List containing paths of all the pages
    sports_list = ['Arsenal_facebook_comments.csv', 'nba_facebook_comments.csv', 'NFL_facebook_comments.csv']

    news_list = ['cnn_facebook_comments.csv', 'FoxNews_facebook_comments.csv', 'TheYoungTurks_facebook_comments.csv']

    entertainment_list = ['BuzzFeed_facebook_comments.csv', '9gag_facebook_comments.csv',
                          'NowThisEntertainment_facebook_comments.csv']

    # Combining data from all pages and randomly sampling 10000 comments.
    return sample_df(sports_list), sample_df(news_list), sample_df(entertainment_list)


def main():
    sports, news, entertainment = merge_fb_data()
    # sports.to_csv("sports_data.csv")
    # news.to_csv("news_data.csv")
    # entertainment.to_csv("entertainment_data.csv")
    sports_preds = pd.read_csv("Sports_Predictions.csv")
    news_preds = pd.read_csv("News_Predictions.csv")
    entertainment_preds = pd.read_csv("Entertainment_Predictions.csv")

    final_df = pd.concat([sports_preds, news_preds, entertainment_preds])
    final_df.to_csv("combined_predictions_data.csv")


if __name__ == "__main__":
    main()