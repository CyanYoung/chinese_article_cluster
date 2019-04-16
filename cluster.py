import json
import pickle as pk

import numpy as np

from preprocess import clean

from util import map_item


path_label = 'data/label.json'
with open(path_label, 'r') as f:
    labels = json.load(f)

topic_num = len(labels)

path_word2ind = 'model/word2ind.pkl'
path_tfidf = 'model/tfidf.pkl'
with open(path_word2ind, 'rb') as f:
    word2ind = pk.load(f)
with open(path_tfidf, 'rb') as f:
    tfidf = pk.load(f)

path_lsi = 'model/lsi.pkl'
path_lda = 'model/lda.pkl'
with open(path_lsi, 'rb') as f:
    lsi = pk.load(f)
with open(path_lda, 'rb') as f:
    lda = pk.load(f)

models = {'lsi': lsi,
          'lda': lda}


def predict(text, name):
    words = clean(text)
    bow_doc = word2ind.doc2bow(words)
    tfidf_doc = tfidf[bow_doc]
    model = map_item(name, models)
    pairs = model[tfidf_doc]
    probs = np.zeros(topic_num)
    for ind, score in pairs:
        probs[ind] = score
    formats = list()
    for prob in probs:
        formats.append('{:.3f}'.format(prob))
    return ', '.join(formats)


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('lsi: %s' % predict(text, 'lsi'))
        print('lda: %s' % predict(text, 'lda'))
