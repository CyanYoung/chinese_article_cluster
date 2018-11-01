import pickle as pk

import numpy as np

from sklearn.cluster import KMeans

from util import map_item


topic_num = 15

path_lsi = 'model/lsi.pkl'
path_lda = 'model/lda.pkl'
with open(path_lsi, 'rb') as f:
    lsi = pk.load(f)
with open(path_lda, 'rb') as f:
    lda = pk.load(f)

feats = {'lsi': lsi,
         'lda': lda}

paths = {'mean_lsi': 'model/mean_lsi.pkl',
         'mean_lda': 'model/mean_lda.pkl'}


def featurize(tfidf_docs, feat):
    topic_docs = list()
    for doc in feat[tfidf_docs]:
        topic_docs.append([score for ind, score in doc])
    return np.array(topic_docs)


def fit(path):
    with open(path, 'rb') as f:
        tfidf_docs = pk.load(f)
    for name, feat in feats.items():
        topic_docs = featurize(tfidf_docs, feat)
        model = KMeans(n_clusters=topic_num, n_init=10, max_iter=100)
        model.fit(topic_docs)
        with open(map_item('mean_' + name, paths), 'wb') as f:
            pk.dump(model, f)


if __name__ == '__main__':
    path = 'feat/tfidf.pkl'
    fit(path)
