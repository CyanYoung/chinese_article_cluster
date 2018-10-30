import json
import pickle as pk

from gensim.models import TfidfModel as Tfidf
from gensim.models import LsiModel as Lsi
from gensim.models import LdaModel as Lda

from gensim.corpora.dictionary import Dictionary

from util import flat_read, map_item


topic_num = 3
key_num = 20

path_word2ind = 'model/word2ind.pkl'
path_tfidf = 'model/tfidf.pkl'

funcs = {'lsi': Lsi,
         'lda': Lda}

paths = {'lsi': 'model/lsi.pkl',
         'lda': 'model/lda.pkl',
         'lsi_feat': 'feat/lsi.json',
         'lda_feat': 'feat/lda.json'}


def save_dict(name, topics):
    topic_pairs = list()
    for ind, all_str in topics:
        pair_strs = all_str.split(' + ')
        pairs = [pair_str.split('*') for pair_str in pair_strs]
        pair_dict = dict()
        for score, key in pairs:
            pair_dict[key[1:-1]] = float(score)
        topic_pairs.append(pair_dict)
    with open(map_item(name + '_feat', paths), 'w') as f:
        json.dump(topic_pairs, f, ensure_ascii=False, indent=4)


def vectorize(path_data, path_vec, mode):
    docs = flat_read(path_data, 'doc')
    doc_words = [doc.split() for doc in docs]
    if mode == 'train':
        word2ind = Dictionary(doc_words)
        bow_docs = [word2ind.doc2bow(words) for words in doc_words]
        tfidf = Tfidf(bow_docs)
        with open(path_word2ind, 'wb') as f:
            pk.dump(word2ind, f)
        with open(path_tfidf, 'wb') as f:
            pk.dump(tfidf, f)
    else:
        with open(path_word2ind, 'rb') as f:
            word2ind = pk.load(f)
        with open(path_tfidf, 'rb') as f:
            tfidf = pk.load(f)
        bow_docs = [word2ind.doc2bow(words) for words in doc_words]
    tfidf_docs = tfidf[bow_docs]
    with open(path_vec, 'wb') as f:
        pk.dump(tfidf_docs, f)
    for name, func in funcs.items():
        model = func(tfidf_docs, id2word=word2ind, num_topics=topic_num)
        topics = model.show_topics(num_words=key_num)
        with open(map_item(name, paths), 'wb') as f:
            pk.dump(model, f)
        if mode == 'train':
            save_dict(name, topics)


if __name__ == '__main__':
    path_data = 'data/train.csv'
    path_vec = 'feat/tfidf_train.pkl'
    vectorize(path_data, path_vec, mode='train')
    path_data = 'data/test.csv'
    path_vec = 'feat/tfidf_test.pkl'
    vectorize(path_data, path_vec, mode='test')
