from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import codecs

def load_documents (file_name):
    list_of_tweets = []
    with codecs.open(file_name, "r", encoding='utf-8', errors='ignore') as harvey_tweets:
        for line in harvey_tweets:
            fields = line.strip('\n')
            list_of_tweets.append(fields)
    return list_of_tweets

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

def nfm_implementation(documents,no_features):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(documents)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = tf_vectorizer.get_feature_names()
    nmf = NMF(n_components=10, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf)
    display_topics(nmf, tf_feature_names, 15)

file_name = '/Users/User/big_data/lda_tweets_1000.csv'
no_of_features = 1000
documents = load_documents(file_name)
print('documents loaded')
nfm_implementation(documents,no_of_features)
