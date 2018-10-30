import os

import jieba
from jieba.posseg import cut as pos_cut

from random import shuffle


path_cut_word = 'dict/cut_word.txt'
jieba.load_userdict(path_cut_word)

pos_set = ('a', 'ad', 'an', 'd', 'n', 'nr', 'ns', 'nt', 'nz', 'v', 'vd', 'vn')


def save(path, docs, labels):
    head = 'label,doc'
    with open(path, 'w') as f:
        f.write(head + '\n')
        for label, doc in zip(labels, docs):
            f.write(label + ',' + doc + '\n')


def filter(text):
    pairs = list(pos_cut(text))
    words = list()
    for word, pos in pairs:
        if pos in pos_set:
            words.append(word)
    return words


def prepare(path_univ_dir, path_train, path_test):
    docs = list()
    labels = list()
    files = os.listdir(path_univ_dir)
    for file in files:
        label = os.path.splitext(file)[0]
        print(label)
        with open(os.path.join(path_univ_dir, file), 'r') as f:
            for line in f:
                words = filter(line.strip())
                cut_doc = ' '.join(words)
                docs.append(cut_doc)
                labels.append(label)
    docs_labels = list(zip(docs, labels))
    shuffle(docs_labels)
    docs, labels = zip(*docs_labels)
    bound = int(len(docs) * 0.9)
    save(path_train, docs[:bound], labels[:bound])
    save(path_test, docs[bound:], labels[bound:])


if __name__ == '__main__':
    path_univ_dir = 'data/univ'
    path_train = 'data/train.csv'
    path_test = 'data/test.csv'
    prepare(path_univ_dir, path_train, path_test)
