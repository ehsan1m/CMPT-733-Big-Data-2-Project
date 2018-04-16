import gensim
import pandas as pd
import Data_Cleaning
import pyLDAvis.gensim

from gensim import corpora, models
from collections import defaultdict


def main():
    #fb_comments = Data_Cleaning.merge_fb_data()
    fb_comments = pd.read_csv("./Final_Predictions/News_Predictions.csv", encoding='ISO-8859-1').sample(n=50000, random_state=10)
    #fb_comments = fb_comments[fb_comments["prediction"] != 0]
    fb_comments["comment_message"] = fb_comments["comment_message"].apply(lambda x: Data_Cleaning.text_cleaning(x).split())
    fb_comments = fb_comments[fb_comments["comment_message"].apply(lambda x: x != [])]

    clean_comms = fb_comments["comment_message"].tolist()
    dictionary = corpora.Dictionary(clean_comms)
    dictionary.save('dictionary.dict')

    dictionary = corpora.Dictionary.load("dictionary.dict")
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in clean_comms]
    corpus_tfidf = models.TfidfModel(doc_term_matrix)[doc_term_matrix]

    lda = gensim.models.LdaModel(corpus_tfidf, num_topics=5, passes=50, id2word=dictionary, minimum_probability=0.1)
    lda.save('topic.model')

    # lda = models.LdaModel.load("topic.model")

    # topic_sizes = defaultdict(int)
    # for doc in clean_comms:
    #     doc_bow = dictionary.doc2bow(doc)
    #     dist = lda[doc_bow]
    #     for topic_size in dist:
    #         topic_id = topic_size[0]
    #         percent = topic_size[1]
    #         topic_sizes[topic_id] += percent
    #
    # print(topic_sizes)
    print(lda.print_topics(num_topics=4, num_words=10))

    corpora.MmCorpus.serialize('news_data.mm', doc_term_matrix)
    c = gensim.corpora.MmCorpus('news_data.mm')

    data = pyLDAvis.gensim.prepare(lda, c, dictionary)
    pyLDAvis.show(data)

    #print(tfidf)
    # for i in range(len(clean_comms)):
    #     print(lda[doc_term_matrix[i]])


if __name__ == "__main__":
    main()