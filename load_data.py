import csv
import numpy as np
import gensim
from gensim.models import Doc2Vec
import pickle

def load_CVAT_2(filename):
    texts, valence, arousal = [], [], []
    with open(filename, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            texts.append(str(line[1]))      # sentence
            valence.append(float(line[2]))        # valence
            arousal.append(float(line[3]))        # arousal

    return texts, valence, arousal


def load_embeddings(arg=None, filename='None'):
    if arg == 'zh_tw':  # dim = 400
        model = gensim.models.Word2Vec.load_word2vec_format(None, binary=False)
    elif arg == 'google_news':
        model = gensim.models.Word2Vec.load_word2vec_format(filename, binary=True)  # C binary format
        w2v = dict()
        for key in model.vocab.keys():
            w2v[key] = model[key]
        return w2v
    elif arg == 'zh':
        model = gensim.models.Word2Vec.load_word2vec_format(filename, binary=False)  # Text format
        w2v = dict()
        for key in model.vocab.keys():
            w2v[key] = model[key]
        return w2v
    elif arg == 'glove':
        glove_path = '/home/hs/Data/Word_Embeddings/glove.840B.300d.txt'
        model = dict()
        with open(glove_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f, quoting=csv.QUOTE_NONE, delimiter=' ')
            for line in reader:
                # print(line[0])
                # split = line.split()
                try:
                    model[line[0]] = np.array(line[1:], dtype=float)
                except:
                    pass
        return model
    elif arg == 'CVAT':  # dim = 50
        model = gensim.models.Word2Vec.load(None)
    elif arg == 'twitter':  # dim = 50
        model = Doc2Vec.load('./data/acc/docvecs_twitter.d2v')
    else:
        raise Exception('Wrong Argument.')
    print('Load Model Complete.')
    return model


def load_pickle(filename):
    out = pickle.load(open(filename, "rb"))
    return out

if __name__=='__main__':
    texts, valence, arousal = load_CVAT_2('../resources/CVAT2.0.csv')
    len_text = []
    for i in texts:
        # print(list(i))
        len_text.append(len(list(i)))
    print(np.mean(np.array(len_text)), np.sum((np.array(len_text))), np.max(np.array(len_text)),
          np.min(np.array(len_text)))
    print(len(texts), len(valence), len(arousal))
