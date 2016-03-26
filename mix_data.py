# encoding: utf-8
from load_data import test_tokenized


def read_index(filename):
    with open(filename, "r") as ins:
        array = []
        for line in ins:
            array.append(int(line))
    return array


def read_specific_line(filename, nb_line):
    # starting from index 0
    out = None
    fp = open(filename, 'r', encoding='utf-8')
    for i, line in enumerate(fp):
        if i == nb_line:
            out = str(line)
            break
    fp.close()
    return out


def read_mix_data(categorical):
    filename1 = './resources/valence_arousal(sigma=1.5).csv'
    tokenized_text_old = test_tokenized('./resources/tokenized_texts_(old).p')
    filename2 = './resources/corpus 2009 sigma 1.5.csv'
    tokenized_text = test_tokenized('./resources/tokenized_texts.p')
    id_list = read_index('./resources/index.txt')
    text_col, label_col, valence_col, arousal_col, = 1, 2, 3, 4
    texts, valence, arousal = [], [], []
    for i in range(2009):
        if i in id_list:
            line = read_specific_line(filename1, i)
            tokenized = tokenized_text_old
        else:
            line = read_specific_line(filename2, i)
            tokenized = tokenized_text

        line = line.split(',')
        if categorical == 'all':
            texts.append(str(tokenized[i]))  # sentence
            valence.append(float(line[valence_col]))  # valence
            arousal.append(float(line[arousal_col]))  # arousal
        elif line[label_col] == categorical:
            texts.append(str(tokenized[i]))  # sentence
            valence.append(float(line[valence_col]))  # valence
            arousal.append(float(line[arousal_col]))  # arousal
    return texts, valence, arousal

if  __name__ == '__main__':
    texts, valence, arousal = read_mix_data('laptop')
    print(texts[-3:])
    print(valence[-3:])
    print(arousal[-3:])
