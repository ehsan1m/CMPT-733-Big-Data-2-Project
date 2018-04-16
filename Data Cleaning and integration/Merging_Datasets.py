import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def LoadDataSets():
    dfSFU = pd.read_csv('Data/SFU_constructiveness_toxicity_corpus.csv')
    dfWiki = pd.read_csv('Data/Wikipedia_train.csv')
    return dfSFU,dfWiki

dfSFU,dfWiki = LoadDataSets()


def PreProcessSFU(dfSFU):
    #Some comments have two ratings. The first one is more popular. Only that one is kept.
    dfSFU['toxicity_level'] = dfSFU['toxicity_level'].apply(lambda x:x.splitlines())
    dfSFU['toxicity_level_1'] = dfSFU['toxicity_level'].apply(lambda x:int(x[0]))
    
#There are also some expert ratings.
#If there is an exper rating available, we use that instead of the crowd's opinion
#Otherwise, the most popular crowd opinion is used.
#Ratings of 1 and 2 are mapped to 0; 3 and 4 to 1
#This is done to create an equivalence between SFU and Wikipedia (more toxic) datasets.
    def mergeRatingsSFU(df):
        if df['expert_toxicity_level'] != df['expert_toxicity_level']:
            rating = df['toxicity_level_1']    
        else:
            rating = df['expert_toxicity_level']

        if (rating == 0 or rating == 1):
            df['merged_rating'] = 0
        else:
            df['merged_rating'] = 1
        return df
    dfSFU = dfSFU.apply(mergeRatingsSFU,axis=1)
    
    # Returning the comment and the final rating columns only
    return dfSFU[['comment_text','merged_rating']]


#Applying preprocessing
dfSFU = PreProcessSFU(dfSFU)


def PreProcessWiki(dfWiki):
#If a comment is severly toxic, threatening, or obscene, it is rated as 2
#Otherwise, if it is toxic or insulting, it is rated as 1
#If non of the above, it is rated as 0

    def mergeRatingsWiki(df):
        if ((df['obscene']==1) or (df['threat']==1) or (df['severe_toxic']==1)):
            df['merged_rating'] = 2
        elif ((df['toxic'] ==1) or (df['insult']==1)):
            df['merged_rating'] = 1
        else:
            df['merged_rating'] = 0
        return df
#     df_merged = dfWiki.iloc[0:1000,:].apply(mergeRatingsWiki,axis=1) #Use this to work with a portion of the data only
    df_merged = dfWiki.apply(mergeRatingsWiki,axis=1)

    
    # Returning the comment and the final rating columns only
    return df_merged[['comment_text','merged_rating']]


#Applying preprocessing
dfWiki = PreProcessWiki(dfWiki)

#Combining the two datasets
dfTrain = pd.concat([dfSFU,dfWiki])

#Saving it as CSV
dfTrain.to_csv('Data/mergedDataSet.csv')


