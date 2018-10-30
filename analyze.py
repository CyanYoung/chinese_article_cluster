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

models = {'lsi': lsi,
          'lda': lda}


def predict(text, name):
    words = filter(text.strip())
    bow_doc = word2ind.doc2bow(words)
    tfidf_doc = tfidf[bow_doc]
    model = map_item(name, models)
    topic_str = ''
    for ind, score in model[tfidf_doc]:
        topic_str = topic_str + '({}, {:.3f}) '.format(ind, score)
    return topic_str


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('lsi: %s' % predict(text, 'lsi'))
        print('lda: %s' % predict(text, 'lda'))
