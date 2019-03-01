import pickle as pk

from gensim.corpora import Dictionary

from gensim.models import TfidfModel as Tfidf

from util import flat_read


path_word2ind = 'model/word2ind.pkl'
path_tfidf = 'model/tfidf.pkl'


def featurize(path_data, path_sent, mode):
    docs = flat_read(path_data, 'doc')
    doc_words = [doc.split() for doc in docs]
    if mode == 'train':
        word2ind = Dictionary(doc_words)
        bow_docs = [word2ind.doc2bow(words) for words in doc_words]
        tfidf = Tfidf(bow_docs)
        with open(path_word2ind, 'wb') as f:
            pk.dump(word2ind, f)
        with open(path_tfidf, 'wb') as f:
            pk.dump(tfidf, f)
    else:
        with open(path_word2ind, 'rb') as f:
            word2ind = pk.load(f)
        with open(path_tfidf, 'rb') as f:
            tfidf = pk.load(f)
        bow_docs = [word2ind.doc2bow(words) for words in doc_words]
    tfidf_docs = tfidf[bow_docs]
    with open(path_sent, 'wb') as f:
        pk.dump(tfidf_docs, f)


if __name__ == '__main__':
    path_data = 'data/train.csv'
    path_sent = 'feat/sent_train.pkl'
    featurize(path_data, path_sent, 'train')
    path_data = 'data/test.csv'
    path_sent = 'feat/sent_test.pkl'
    featurize(path_data, path_sent, 'test')
