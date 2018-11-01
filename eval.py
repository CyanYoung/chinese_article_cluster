import pickle as pk

import numpy as np

from build import featurize

from util import flat_read, map_item


topic_num = 15

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


def label2ind(labels):
    label_set = sorted(list(set(labels)))
    label_inds = dict()
    for i in range(len(label_set)):
        label_inds[label_set[i]] = i
    inds = [label_inds[label] for label in labels]
    return np.array(inds)


def test(tfidf_docs, labels):
    labels = label2ind(labels)
    for name, model in models.items():
        feat = map_item(name, feats)
        topic_docs = featurize(tfidf_docs, feat)
        preds = model.predict(topic_docs)
        accs = list()
        for i in range(topic_num):
            pred_inds = np.where(preds == i)
            match_labels = labels[pred_inds]
            counts = np.bincount(match_labels)
            accs.append(np.max(counts) / np.sum(counts))
        print('\n%s %s %.2f' % (name, 'acc:', sum(accs) / len(accs)))


if __name__ == '__main__':
    test(tfidf_docs, labels)
