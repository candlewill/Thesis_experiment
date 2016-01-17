import numpy as np
from metrics import continuous_metrics
from visualize import draw_linear_regression
from visualize import plot_keras, draw_hist


def regression_evaluate(y_test, predict, msg='None'):  # msg: Valence or Arousal

    continuous_metrics(y_test, predict, 'prediction result:')

    # visualization


    X = range(50, 100)  # or range(len(y_test))
    draw_linear_regression(X, np.array(y_test)[X], np.array(predict)[X], 'Sentence Number', msg,
                           'Comparison of predicted and true ' + msg)

    draw_hist(np.array(y_test) - np.array(predict), title='Histogram of ' + msg + ' prediction: ')
