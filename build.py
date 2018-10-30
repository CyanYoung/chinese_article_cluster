import pickle as pk

import numpy as np

from sklearn.cluster import KMeans


topic_num = 15

path_lsi = 'model/lsi.pkl'
path_lda = 'model/lda.pkl'
with open(path_lsi, 'rb') as f:
    lsi = pk.load(f)
with open(path_lda, 'rb') as f:
    lda = pk.load(f)

feats = {'lsi': lsi,
         'lda': lda}


def fit(path):
    with open(path, 'rb') as f:
        tfidf_docs = pk.load(f)
    model = KMeans(n_clusters=topic_num, n_init=10, max_iter=100)
    for name, feat in feats.items():
        doc_topics = np.array(feat[tfidf_docs])
        model.fit(doc_topics)
        print(model.labels_)


if __name__ == '__main__':
    path = 'feat/tfidf.pkl'
    fit(path)
