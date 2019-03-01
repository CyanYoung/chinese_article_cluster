import pickle as pk

import numpy as np

from cluster import models

from util import map_item


path_test = 'feat/sent_test.pkl'
with open(path_test, 'rb') as f:
    sents = pk.load(f)


def test(name, sents):
    model = map_item(name, models)
    log = model.log_perplexity(sents)
    print('\n%s %s %.2f' % (name, 'perp:', np.power(2, -log)))


if __name__ == '__main__':
    test('lda', sents)
