
# coding: utf-8

# # Importing Libraries

# In[1]:


# Importing Libraries
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()


# # Combining data from multiple pages

# In[7]:


# Function to read comments file of different pages
def read_file(file_path):
    df = pd.read_csv(file_path)
    return df

# List containing paths of all the pages
page_list = ['cnn_facebook_comments.csv','FoxNews_facebook_comments.csv','TheYoungTurks_facebook_comments.csv',            'BuzzFeed_facebook_comments.csv','9gag_facebook_comments.csv','NowThisEntertainment_facebook_comments.csv',            'Arsenal_facebook_comments.csv','nba_facebook_comments.csv','NFL_facebook_comments.csv']

# Combining data from all pages and randomly sampling 10000 comments.
chunks =[]
for page in page_list:
    df_page = read_file(page)
    chunks.append(df_page)
    df_concat = pd.concat(chunks,ignore_index=True)
df_concat = df_concat['comment_message'].sample(n=10000,random_state =10)


# # Function to Clean Data

# In[8]:


def text_cleaning(comment):
    comment = comment.lower()                               # Converting comments into lowercase
    comment = comment.strip("b'").strip('b"')               # Removing b",b' from start and end of comment 
    comment = re.sub("\\[\\[(.*?)\\]\\]","",comment)        # Removing GIFs and images from comments
    comment = re.sub(r'\\x\S+',"",comment)                  # Removing unwanted text , emojis etc.
#   Replacing apostrophes
    comment = re.sub(r"'s",' is',comment)                   
    comment = re.sub(r"'re",' are',comment)
    comment = re.sub(r"'t",' not',comment)
    comment = re.sub(r"'m",' am',comment)
    comment = re.sub(r"'d",' would',comment)
    comment = re.sub(r"'ll",' will',comment)
    comment = re.sub(r"'ve",' have',comment)
    comment = re.sub('[.]', ' ', comment)
    
    comment = ''.join([c for c in comment if c not in ('!', '?' ,'.','\\','"',',','-','$','%',"'")]) #Removing Punctuations and other signs
    comment = re.sub(r'[0-9]',"",comment)                    # Removing numbers
    comment = re.sub(r'http\S+',"",comment)                  # Removing Url
    comment = ' '.join([c for c in comment.split() if c not in stop_words])   # Removing stopwords
    comment = ' '.join([lemmatizer.lemmatize(c) for c in comment.split()])    # Lemmatizing
    return comment
df_concat = df_concat.apply(lambda x: text_cleaning(x))
df_concat = df_concat[df_concat.apply(lambda x: x is not "")]

