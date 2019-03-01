import json
import pickle as pk

from gensim.models import LsiModel as Lsi
from gensim.models import LdaModel as Lda

from util import map_item


key_num = 20

path_label = 'data/label.json'
path_word2ind = 'model/word2ind.pkl'
with open(path_label, 'r') as f:
    labels = json.load(f)
with open(path_word2ind, 'rb') as f:
    word2ind = pk.load(f)

topic_num = len(labels)

funcs = {'lsi': Lsi,
         'lda': Lda}

paths = {'lsi_dict': 'dict/lsi.json',
         'lda_dict': 'dict/lda.json',
         'lsi': 'model/lsi.pkl',
         'lda': 'model/lda.pkl'}


def save_dict(name, topics):
    topic_pairs = list()
    for ind, all_str in topics:
        pair_strs = all_str.split(' + ')
        pairs = [pair_str.split('*') for pair_str in pair_strs]
        pair_dict = dict()
        for score, key in pairs:
            pair_dict[key[1:-1]] = float(score)
        topic_pairs.append(pair_dict)
    with open(map_item(name + '_dict', paths), 'w') as f:
        json.dump(topic_pairs, f, ensure_ascii=False, indent=4)


def fit(path_train):
    with open(path_train, 'rb') as f:
        sents = pk.load(f)
    for name, func in funcs.items():
        model = func(sents, id2word=word2ind, num_topics=topic_num)
        topics = model.show_topics(num_words=key_num)
        save_dict(name, topics)
        with open(map_item(name, paths), 'wb') as f:
            pk.dump(model, f)


if __name__ == '__main__':
    path_train = 'feat/sent_train.pkl'
    fit(path_train)
