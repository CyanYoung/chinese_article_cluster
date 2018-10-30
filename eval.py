import pickle as pk

from sklearn.metrics import accuracy_score

from build import featurize

from util import flat_read, map_item


path_test = 'data/test.csv'
path_vec = 'feat/tfidf_test.pkl'
labels = flat_read(path_test, 'label')
with open(path_vec, 'rb') as f:
    tfidf_docs = pk.load(f)

path_lsi = 'model/lsi.pkl'
path_lda = 'model/lda.pkl'
path_mean_lsi = 'model/mean_lsi.pkl'
path_mean_lda = 'model/mean_lda.pkl'
with open(path_lsi, 'rb') as f:
    lsi = pk.load(f)
with open(path_lda, 'rb') as f:
    lda = pk.load(f)
with open(path_mean_lsi, 'rb') as f:
    mean_lsi = pk.load(f)
with open(path_mean_lda, 'rb') as f:
    mean_lda = pk.load(f)

feats = {'lsi': lsi,
         'lda': lda}

models = {'lsi': mean_lsi,
          'lda': mean_lda}


def test(tfidf_docs, labels):
    for name, model in models.items():
        feat = map_item(name, feats)
        topic_docs = featurize(tfidf_docs, feat)
        preds = model.predict(topic_docs)



if __name__ == '__main__':
    test(tfidf_docs, labels)
