import matplotlib.pyplot as plt
import numpy as np
from statistics import mean


def draw_scatter(x, y, x_labels, y_labels, title='CVAT 2.0 VA Scatter'):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(x, y, marker='o', color='#78A5A3')
    plt.axhline(mean(y), color='#CE5A57')
    plt.axvline(mean(x), color='#CE5A57')
    plt.xlabel(x_labels)
    plt.ylabel(y_labels)
    plt.title(title)
    plt.show()


def draw_linear_regression(X, Y_test, predict, x_labels, y_labels, title):
    plt.plot(X, Y_test, color='#78A5A3', marker='o', label='True Value')
    plt.plot(X, predict, color='#CE5A57', marker='D', label='Predicted Value')

    plt.xlabel(x_labels)
    plt.ylabel(y_labels)
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()


def draw_iter(train_loss, valid_loss, x_labels, y_labels):
    plt.plot(train_loss, linewidth=3, label="Train Loss")
    plt.plot(valid_loss, linewidth=3, label="Val Loss")
    plt.grid()
    plt.legend()
    plt.xlabel(x_labels)
    plt.xlim(xmin=0)
    # plt.ylim(ymin=0)
    plt.ylabel(y_labels)
    plt.show()


def plot_keras(result, acc=False, x_labels=None, y_labels=None):
    # plot the result
    if acc == True:
        plt.figure
        plt.plot(result.epoch, result.history['acc'], label="acc")
        plt.plot(result.epoch, result.history['val_acc'], label="val_acc")
        plt.scatter(result.epoch, result.history['acc'], marker='*')
        plt.scatter(result.epoch, result.history['val_acc'])
        plt.legend(loc='under right')
        plt.show()

    plt.figure
    plt.plot(result.epoch, result.history['loss'], label="Train Loss", marker='D', color='#CE5A57', linewidth=3)
    plt.plot(result.epoch, result.history['val_loss'], label="Val Loss", marker='o', color='#78A5A3', linewidth=3)
    plt.grid()
    plt.xlabel(x_labels)
    # plt.xlim(xmin=1)
    # plt.ylim(ymin=0)
    plt.ylabel(y_labels)
    plt.legend(loc='upper right')
    plt.show()


def draw_hist(data, title):
    plt.figure
    import matplotlib.mlab as mlab
    # example data
    mu = np.mean(data)  # mean of distribution
    sigma = np.std((data))  # standard deviation of distribution

    num_bins = 20
    # the histogram of the data
    n, bins, patches = plt.hist(data, num_bins, normed=1, facecolor='#78A5A3')
    # add a 'best fit' line
    y = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, y, '--', color="#CE5A57")
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title(title + r'$\mu=%.3f$, $\sigma=%.3f$' % (mu, sigma))

    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.grid(True)
    plt.show()


# draw_hist(np.array([2,3,2,3,1,5,4,3,2,3,2,3,2,1,4,2,3,5,2]),'few')
# exit()


if __name__ == '__main__':
    from load_data import load_CVAT_2

    texts, valence, arousal = load_CVAT_2('./resources/CVAT2.0(sigma=1.0).csv')
    draw_scatter(valence, arousal, 'Valence', 'Arousal')
