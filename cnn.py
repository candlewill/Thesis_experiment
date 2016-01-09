from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedMerge
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.datasets import imdb
from keras.constraints import unitnorm
from keras.layers.core import Reshape, Flatten, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from VA_prediction import build_keras_input
import numpy as np
from sklearn import cross_validation


def cnn(W=None):
    # Number of feature maps (outputs of convolutional layer)
    N_fm = 20
    dense_nb = 20
    # kernel size of convolutional layer
    kernel_size = 5
    conv_input_width = W.shape[1]  # dims=300
    global maxlen
    conv_input_height = maxlen  # maxlen of sentence

    model = Sequential()
    # Embedding layer (lookup table of trainable word vectors)
    model.add(Embedding(input_dim=W.shape[0], output_dim=W.shape[1], weights=[W], W_constraint=unitnorm()))
    # Reshape word vectors from Embedding to tensor format suitable for Convolutional layer
    model.add(Reshape(dims=(1, conv_input_height, conv_input_width)))

    # first convolutional layer
    model.add(Convolution2D(nb_filter=N_fm, nb_row=kernel_size, nb_col=conv_input_width, border_mode='valid',
                            W_regularizer=l2(0.0001), activation='relu'))
    # ReLU activation
    model.add(Dropout(0.5))

    # aggregate data in every feature map to scalar using MAX operation
    # model.add(MaxPooling2D(pool_size=(conv_input_height-kernel_size+1, 1), border_mode='valid'))
    model.add(MaxPooling2D(pool_size=(kernel_size, 1), border_mode='valid'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(output_dim=dense_nb, activation='relu'))
    model.add(Dropout(0.5))
    # Inner Product layer (as in regular neural network, but without non-linear activation function)
    model.add(Dense(output_dim=1, activation='linear'))
    # SoftMax activation; actually, Dense+SoftMax works as Multinomial Logistic Regression
    return model


if __name__ == '__main__':
    (data, W) = build_keras_input()
    (idx_data, valence, arousal) = data

    maxlen = 100  # cut texts after this number of words (among top max_features most common words)
    batch_size = 16

    option = 'valence'  # or arousal
    Y = np.array(valence) if option == 'Valence' else np.array(arousal)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(idx_data, Y, test_size=0.2,
                                                                         random_state=0)

    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')
    # m= 0
    # for i in X_train:
    #     if len(i) >0:
    #         for j in i:
    #             if j > m:
    #                 m=j
    # print(m)
    max_features = W.shape[0]  # shape of W: (13631, 300) , changed to 14027 through min_df = 3

    print("Pad sequences (samples x time)")
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    model = cnn(W)

    model.compile(loss='mean_squared_error', optimizer='adagrad')  # loss function: mse

    print("Train...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=20, validation_data=(X_test, y_test),
              callbacks=[early_stopping])
    score = model.evaluate(X_test, y_test, batch_size=batch_size)
    print('Test score:', score)