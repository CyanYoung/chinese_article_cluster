import pickle as pk

from preprocess import filter

from util import map_item


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

feats = {'lsi': lsi,
         'lda': lda}


def predict(text, name):
    words = filter(text.strip())
    bow_doc = word2ind.doc2bow(words)
    tfidf_doc = tfidf[bow_doc]
    feat = map_item(name, feats)
    formats = list()
    for ind, score in feat[tfidf_doc]:
        formats.append('({}, {:.3f})'.format(ind, score))
    return ' '.join(formats)


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('lsi: %s' % predict(text, 'lsi'))
        print('lda: %s' % predict(text, 'lda'))
