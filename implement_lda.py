import logging
import codecs
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models, similarities
import gensim
import warnings
from gensim.models import LdaModel
warnings.filterwarnings('ignore')
import pyLDAvis.gensim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

file_name = '/Users/User/big_data/lda_tweets_1000.csv'
def load_documents (file_name):
    list_of_tweets = []
    with codecs.open(file_name, "r", encoding='utf-8', errors='ignore') as harvey_tweets:
        for line in harvey_tweets:
            fields = line.strip('\n')
            list_of_tweets.append(fields)
    return list_of_tweets

def tokenization(doc):
    tokenizer = RegexpTokenizer(r'\w+')
    raw = doc.lower()
    tokens = tokenizer.tokenize(raw)
    return tokens

def remove_stop_words(list_of_tokens):
    tokens_without_stop_words = []
    stop_words = set(stopwords.words('english'))
    for token in list_of_tokens:
        if token not in stop_words:
            tokens_without_stop_words.append(token)
    return tokens_without_stop_words

def stemming_words(tokens_without_stop_words):
    p_stemmer = PorterStemmer()
    list_of_stemmed_tokens = [p_stemmer.stem(i) for i in tokens_without_stop_words]
    return list_of_stemmed_tokens

def construct_document_term_matrix(list_of_tokenized_tweets):
    dictionary = corpora.Dictionary(list_of_tokenized_tweets)
    dictionary.save('dictionary.dict')


    document_term_matrix = [dictionary.doc2bow(text) for text in list_of_tokenized_tweets]
    corpora.MmCorpus.serialize('corpus.mm', document_term_matrix)

    return document_term_matrix,dictionary

def lda(file_name):
    tweet_count = 0
    list_of_tweets = load_documents(file_name)
    print('done loading documents')
    list_of_tokenized_tweets = []
    for tweet in list_of_tweets:
        tweet_count+= 1
        if (tweet_count %1000) == 0:
            print('tweet count ---- >',tweet_count)
        list_of_tokens = tokenization(tweet)
        tokens_without_stop_words = remove_stop_words(list_of_tokens)
        list_of_stemmed_tokens = stemming_words(tokens_without_stop_words)
        list_of_tokenized_tweets.append(list_of_stemmed_tokens)
    matrix_dict = construct_document_term_matrix(list_of_tokenized_tweets)
    print('done constructing matrix')
    document_term_matrix = matrix_dict[0]
    dictionary = matrix_dict[1]
    print('started calculating lda')
    #generate lda model
    Lda = gensim.models.ldamodel.LdaModel

    ldamodel = Lda(document_term_matrix, num_topics=10, id2word=dictionary, passes=20)
    ldamodel.save('topic.model')
    loading = LdaModel.load('topic.model')


    print('done calculating lda')
    return (ldamodel)




ldamodel = lda(file_name)
loading = LdaModel.load('topic.model')

print(loading.print_topics(num_topics=10, num_words=20))


d = gensim.corpora.Dictionary.load('dictionary.dict')
c = gensim.corpora.MmCorpus('corpus.mm')
lda1 = gensim.models.LdaModel.load('topic.model')
data = pyLDAvis.gensim.prepare(lda1, c, d)
pyLDAvis.save_html(data, '/Users/User/Desktop/LDA_Visualization.html')
pyLDAvis.save_html(data, '/Users/User/PycharmProjects/harvey_tweets/lda_harvery/LDA_Visualization.html')

'''
#another implementation of lda
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

def another_implementation_lda(documents):
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = tf_vectorizer.get_feature_names()
    lda = LatentDirichletAllocation(n_topics=10, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
    display_topics(lda, tf_feature_names, 15)

#documents = load_documents(file_name)
#another_implementation_lda(documents)
'''
