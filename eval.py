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
path_km_lsi = 'model/km_lsi.pkl'
path_km_lda = 'model/km_lda.pkl'
with open(path_lsi, 'rb') as f:
    lsi = pk.load(f)
with open(path_lda, 'rb') as f:
    lda = pk.load(f)
with open(path_km_lsi, 'rb') as f:
    km_lsi = pk.load(f)
with open(path_km_lda, 'rb') as f:
    km_lda = pk.load(f)

feats = {'lsi': lsi,
         'lda': lda}

models = {'km_lsi': km_lsi,
          'km_lda': km_lda}


def label2ind(labels):
    labels = sorted(list(set(labels)))
    label_inds = dict()
    for i in range(len(labels)):
        label_inds[labels[i]] = i
    return label_inds


def ind2label(label_inds):
    ind_labels = dict()
    for label, ind in label_inds.items():
        ind_labels[ind] = label
    return ind_labels


def test(tfidf_docs, labels):
    label_inds = label2ind(labels)
    inds = np.array([label_inds[label] for label in labels])
    ind_labels = ind2label(label_inds)
    for name, feat in feats.items():
        model = map_item('km_' + name, models)
        topic_docs = featurize(tfidf_docs, feat)
        preds = model.predict(topic_docs)
        max_labels = list()
        rates = list()
        for i in range(topic_num):
            pred_args = np.where(preds == i)
            match_inds = inds[pred_args]
            counts = np.bincount(match_inds)
            max_labels.append(ind_labels[np.argmax(counts)])
            rates.append(max(counts) / sum(counts))
        print('\n%s %s %.2f\n' % (name, 'acc:', sum(rates) / len(rates)))
        formats = list()
        for label, rate in zip(max_labels, rates):
            formats.append('{} {:.2f}'.format(label, rate))
        print(', '.join(formats))


if __name__ == '__main__':
    test(tfidf_docs, labels)
