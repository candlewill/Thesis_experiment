import matplotlib.pyplot as plt
import numpy as np
from statistics import mean


def draw_scatter(x, y, x_labels, y_labels, title='CVAT 2.0 VA Scatter'):

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(x, y, marker='o')
    plt.axhline(mean(y), color='black')
    plt.axvline(mean(x), color='black')
    plt.xlabel(x_labels)
    plt.ylabel(y_labels)
    plt.title(title)
    plt.show()

# def draw


if __name__=='__main__':
    from load_data import load_CVAT_2
    texts, valence, arousal = load_CVAT_2('./resources/CVAT2.0.csv')
    draw_scatter(valence, arousal, 'Valence', 'Arousal')