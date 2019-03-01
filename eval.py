import pickle as pk

import numpy as np

from util import map_item


path_test = 'feat/sent_test.pkl'
with open(path_test, 'rb') as f:
    sents = pk.load(f)

path_lsi = 'model/lsi.pkl'
path_lda = 'model/lda.pkl'
with open(path_lsi, 'rb') as f:
    lsi = pk.load(f)
with open(path_lda, 'rb') as f:
    lda = pk.load(f)

models = {'lsi': lsi,
          'lda': lda}


def test(name, sents):
    model = map_item(name, models)
    log = model.log_perplexity(sents)
    print('\n%s %s %.2f' % (name, 'perp:', np.power(2, -log)))


if __name__ == '__main__':
    test('lda', sents)
