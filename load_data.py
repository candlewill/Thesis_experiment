import csv
import numpy as np
import gensim
from gensim.models import Doc2Vec
import pickle
import shutil
import hashlib
from sys import platform
import os
import codecs


def load_CVAT_2(filename, categorical='all'):
    texts, valence, arousal = [], [], []
    # without texts categorical information
    # text_col, valence_col, arousal_col = 1, 2, 3
    # with categorical information
    text_col, label_col, valence_col, arousal_col, = 1, 2, 3, 4
    with open(filename, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        if categorical == 'all':
            for line in reader:
                texts.append(str(line[text_col]))  # sentence
                valence.append(float(line[valence_col]))  # valence
                arousal.append(float(line[arousal_col]))  # arousal
        elif categorical in ["book", "car", "laptop", "hotel", "news", "political"]:
            for line in reader:
                if line[label_col] == categorical:
                    texts.append(str(line[text_col]))  # sentence
                    valence.append(float(line[valence_col]))  # valence
                    arousal.append(float(line[arousal_col]))  # arousal
        else:
            raise Exception("Parameters Wrong: categorical not exist.")

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
        glove_path = '/home/hs/Data/wikipedia/GloVe/Traditional_word/vectors.txt'
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


def load_GloVe(filename, num_lines, dims):
    def prepend_line(infile, outfile, line):
        """
        Function use to prepend lines using bash utilities in Linux.
        (source: http://stackoverflow.com/a/10850588/610569)
        """
        with open(infile, 'r') as old:
            with open(outfile, 'w') as new:
                new.write(str(line) + "\n")
                shutil.copyfileobj(old, new)

    def prepend_slow(infile, outfile, line):
        """
        Slower way to prepend the line by re-creating the inputfile.
        """
        with open(infile, 'r') as fin:
            with open(outfile, 'w') as fout:
                fout.write(line + "\n")
                for line in fin:
                    fout.write(line)

    gensim_file = filename + '.w2v_format'
    gensim_first_line = "{} {}".format(num_lines, dims)
    # Prepends the line.

    if platform == "linux" or platform == "linux2":
        prepend_line(filename, gensim_file, gensim_first_line)
    else:
        prepend_slow(filename, gensim_file, gensim_first_line)

    model = gensim.models.Word2Vec.load_word2vec_format(gensim_file, binary=False)  # GloVe Model
    print('GloVe model loaded.')

    return model


def load_pickle(filename):
    out = pickle.load(open(filename, "rb"))
    return out


def load_CVAW(extended=False):
    lexicon_data = []
    filename = './resources/CVAW.txt'
    fr = codecs.open(filename, 'r', 'utf-8')
    for line in fr.readlines():
        line = line.strip().split(',')
        lexicon_data.append([line[0], float(line[1]), float(line[2])])

    if extended == True:
        extended_filename = './resources/neural_cand.txt'
        for line in codecs.open(extended_filename, 'r', 'utf-8').readlines():
            line = line.strip().split()
            lexicon_data.append([line[0], float(line[1]), float(line[2])])

    return lexicon_data

def test_tokenized(tokenized_texts_filename):
    texts = load_pickle(tokenized_texts_filename)
    out = []
    for i in texts:
        out.append(" ".join([w.replace(" ", "") for w in i.split("   ")]))
    return out

def save_to_foler(ls):
    for i, text in enumerate(ls):
        text_file = open("./tmp/tokenized/"+str(i+1)+".txt", "w", encoding='utf-8')
        text_file.write(text)
        text_file.close()
# save the tokenized data into a folder
# save_to_foler(test_tokenized())
# exit()

def load_CVAT_3(file_name, tokenized_texts_filename, categorical):
    # categorical values: "all", "book", "car", "laptop", "hotel", "news", "political"
    # texts, valence, arousal = load_CVAT_2('./resources/corpus 2009 sigma 1.5.csv', categorical="all")
    texts, valence, arousal = load_CVAT_2(file_name, categorical="all")
    tokenized_texts = test_tokenized(tokenized_texts_filename)
    if categorical == "all":
        pass
    elif categorical in ["book", "car", "laptop", "hotel", "news", "political"]:
        texts, valence, arousal = [], [], []
        text_col, label_col, valence_col, arousal_col, = 1, 2, 3, 4
        with open(file_name, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            for i,line in enumerate(reader):
                if line[label_col] == categorical:
                    texts.append(str(tokenized_texts[i]))  # sentence
                    valence.append(float(line[valence_col]))  # valence
                    arousal.append(float(line[arousal_col]))  # arousal
            tokenized_texts = texts
    else:
        raise Exception("Parameters Wrong: categorical not exist.")
    return tokenized_texts, valence, arousal


if __name__ == '__main__':
    texts, valence, arousal = load_CVAT_3('./resources/corpus 2009 sigma 1.5.csv','./resources/tokenized_texts.p', categorical="political")
    # texts = load_pickle("./resources/tokenized_texts.p")
    len_text = []
    from CKIP_tokenizer import segsentence
    out = []
    for idx, i in enumerate(texts):
        # print(list(i))
        # print(idx)
        # out.append(" ".join(segsentence(i)))
        len_text.append(len(i.split()))
    # from save_data import dump_picle
    # dump_picle(out, "tokenized_texts.p")
    # print("The tokenized text is saved.")
    # exit()
    print(np.mean(np.array(len_text)), np.sum((np.array(len_text))), np.max(np.array(len_text)),
          np.min(np.array(len_text)))
    print(len(texts), len(valence), len(arousal))
    print(texts[:20])
