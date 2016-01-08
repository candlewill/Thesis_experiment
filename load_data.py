import csv
import numpy as np

def load_CVAT_2(filename):
    texts, valence, arousal = [], [], []
    with open(filename, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            texts.append(str(line[1]))      # sentence
            valence.append(float(line[2]))        # valence
            arousal.append(float(line[3]))        # arousal

    return texts, valence, arousal

if __name__=='__main__':
    texts, valence, arousal = load_CVAT_2('../resources/CVAT2.0.csv')
    len_text = []
    for i in texts:
        # print(list(i))
        len_text.append(len(list(i)))
    print(np.mean(np.array(len_text)), np.sum((np.array(len_text))))
    print(len(texts), len(valence), len(arousal))
