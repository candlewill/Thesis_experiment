# encoding=utf-8
import jieba
from load_data import load_CVAT_2
from load_data import load_embeddings
from save_data import dump_picle
from load_data import load_pickle
import numpy as np
from collections import defaultdict
import os
import itertools, unicodedata
import re


def clean_str_character(s):
    # This is a closure for key(), encapsulated in an array to work around
    # 2.x's lack of the nonlocal keyword.
    sequence = [0x10000000]

    def key(part):
        val = ord(part)
        if part.isspace():
            return 0

        # This is incorrect, but serves this example; finding a more
        # accurate categorization of characters is up to the user.
        asian = unicodedata.category(part) == "Lo"
        if asian:
            # Never group asian characters, by returning a unique value for each one.
            sequence[0] += 1
            return sequence[0]

        return 2

    result = []
    for key, group in itertools.groupby(s, key):
        # Discard groups of whitespace.
        if key == 0:
            continue

        str = "".join(group)
        result.append(str)

    return " ".join(result)


def clean_str_word(s):
    seg_list = jieba.cut(s, cut_all=False)
    sent = " ".join(seg_list)
    return re.sub(r" +", " ", sent).strip()  # remove additional blank space


def clean_str(s, level='word'):
    if level == 'word':
        out = clean_str_word(s)
    elif level == 'character':
        out = clean_str_character(s)
    else:
        raise Exception('Argument Error! use word or character only.')
    return out


# Test clean_str
# print(clean_str('我是一个好人， 你呢？ my name is Yunchao很高兴认识你'))
# exit()

# return the vocabulary dictionary, format: word-frequency
def get_vocab(corpus):
    vocab = defaultdict(int)
    for sent in corpus:
        for word in clean_str(sent).split():
            vocab[word] += 1
    print('The total number of vocabulary is: %s. ' % len(vocab))
    return vocab


def add_unknown_words(word_vecs, vocab, min_df=3, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)
    return word_vecs


# word_vecs is the model of word2vec
def build_embedding_matrix(word_vecs, vocab, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    union = (set(word_vecs.keys()) & set(vocab.keys()))
    vocab_size = len(union)
    print('The number of words occuring in corpus and word2vec simutaneously: %s.' % vocab_size)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size + 1, k))
    W[0] = np.zeros(k, dtype=np.float32)
    for i, word in enumerate(union, start=1):
        print(word, i)
        W[i] = word_vecs[word]
        word_idx_map[word] = i  # dict
    return W, word_idx_map


def sent2ind(sent, word_idx_map):
    """
    Transforms sentence into a list of indices.
    """
    x = []
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    return x


def make_idx_data(sentences, word_idx_map):
    """
    Transforms sentences (corpus, a list of sentence) into a 2-d matrix.
    """
    idx_data = []
    for sent in sentences:
        idx_sent = sent2ind(clean_str(sent), word_idx_map)
        idx_data.append(idx_sent)
    # idx_data = np.array(idx_data, dtype=np.int)
    return idx_data


def build_keras_input():
    filename_data, filename_w = './tmp/indexed_data_glove.p', './tmp/Weight_glove.p'

    if os.path.isfile(filename_data) and os.path.isfile(filename_w):
        data = load_pickle(filename_data)
        W = load_pickle(filename_w)
        print('Load OK.')
        return (data, W)

    # load data from pickle
    texts, valence, arousal = load_CVAT_2('./resources/CVAT2.0.csv')

    vocab = get_vocab(texts)
    # word_vecs = load_embeddings('google_news', '/home/hs/Data/Word_Embeddings/google_news.bin')
    # word_vecs = load_embeddings('zh',
    #                             '/home/hs/Data/wikipedia/word2vec_word/traditional_wordvecs/wiki.zh.text.traditional_wordvecs.txt')
    # load glove vectors
    word_vecs = load_embeddings(arg='glove')

    word_vecs = add_unknown_words(word_vecs, vocab)
    W, word_idx_map = build_embedding_matrix(word_vecs, vocab)

    idx_data = make_idx_data(texts, word_idx_map)

    data = (idx_data, valence, arousal)

    dump_picle(data, filename_data)
    dump_picle(W, filename_w)
    return (data, W)


if __name__ == '__main__':
    build_keras_input()
