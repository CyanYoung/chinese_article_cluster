import pickle as pk

from sklearn.metrics import accuracy_score

from util import flat_read


path_test = 'data/test.csv'
path_feat = 'feat/tfidf_test.pkl'
labels = flat_read(path_test, 'label')
with open(path_feat, 'rb') as f:
    tfidf_docs = pk.load(f)

path_lsi = 'model/mean_lsi.pkl'
path_lda = 'model/mean_lda.pkl'
with open(path_lsi, 'rb') as f:
    mean_lsi = pk.load(f)
with open(path_lda, 'rb') as f:
    mean_lda = pk.load(f)

models = {'mean_lsi': mean_lsi,
          'mean_lda': mean_lda}


def test(tfidf_docs, labels):
    pass


if __name__ == '__main__':
    test(tfidf_docs, labels)
