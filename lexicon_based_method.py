# coding: utf-8
from load_data import load_CVAT_2
from load_data import load_CVAW
from VA_prediction import clean_str_word
from CKIP_tokenizer import segsentence
from evaluate import regression_evaluate
from sklearn import cross_validation
import numpy as np
from sklearn import linear_model
from visualize import draw_scatter


def mean_ratings(texts, lexicon, mean_method, true_values):
    predicted_ratings = []
    global tokenizer

    def tf_geo(text):  # tf_geo
        sum_valence = 1
        count = 0
        word_list = text.split()
        for word in word_list:
            for line in lexicon:
                if word == line:  # original is ==
                    count = count + 1
                    sum_valence = sum_valence * lexicon[line]
        if count == 0:
            print(text)
        return 5 if count == 0 else sum_valence ** (1. / count)  # geo

    def tf_mean(text):  # tf_mean
        print("Sentence:")
        print(text)
        sum_valence = 0
        count = 0
        word_list = text.split()
        for word in word_list:
            for line in lexicon:
                if word == line:
                    print('The sentiment words:')
                    print("word:%s, arousal: %s." % (word, lexicon[word]))
                    count = count + 1
                    sum_valence = sum_valence + lexicon[line]
        predicted_value = 5 if count == 0 else sum_valence / count
        print('Predicted value: %s' % predicted_value)
        return predicted_value

    if mean_method == 'tf_geo':
        VA_mean = tf_geo
    elif mean_method == 'tf_mean':
        VA_mean = tf_mean
    else:
        raise Exception('Parameters Wrong.')

    for i, text in enumerate(texts):
        # print(text)
        V = VA_mean(tokenizer(text))
        print("True value: %s" % true_values[i])
        predicted_ratings.append(V)
    print(predicted_ratings[:200])
    return predicted_ratings


def linear_regression(X_train, X_test, Y_train, Y_test, plot=False):
    # Create linear regression object
    # The training data should be column vectors
    X_train, X_test = np.array(X_train).reshape((len(X_train), 1)), np.array(X_test).reshape((len(X_test), 1))
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    regr.fit(X_train, Y_train)
    predict = regr.predict(X_test)
    return regression_evaluate(Y_test, predict)


def cv(data, target):
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(data, target, test_size=0.2, random_state=2)
    return linear_regression(X_train, X_test, Y_train, Y_test, plot=False)


def split(sentence):
    return sentence


if __name__ == '__main__':

    ####################### Hyper-parameters #########################
    using_extended_lexicon = False  # 'True' or 'False'
    option = 'A'  # 'V' or 'A'
    mean_method = 'tf_mean'  # values: 'tf_geo', 'tf_mean'
    sigma = 1.5  # values: '1.0', '1.5', '2.0'
    tokenizer = 'pre_tokenized'  # values: 'jieba', 'ckip', "pre_tokenized"
    categorical = 'laptop'  # values: 'all', "book", "car", "laptop", "hotel", "news", "political"
    ##################################################################

    if tokenizer == 'ckip':
        tokenizer = segsentence
    elif tokenizer == 'jieba':
        tokenizer = clean_str_word
    elif tokenizer == "pre_tokenized":
        tokenizer = split

    # texts, valence, arousal = load_CVAT_2('./resources/CVAT2.0(sigma=' + str(sigma) + ').csv', categorical=categorical)
    # texts, valence, arousal = load_CVAT_2("./resources/valence_arousal(sigma=1.5).csv", categorical=categorical)
    from load_data import load_CVAT_3
    # texts, valence, arousal = load_CVAT_3('./resources/corpus 2009 sigma 1.5.csv','./resources/tokenized_texts.p', categorical=categorical)
    # texts, valence, arousal = load_CVAT_3('./resources/valence_arousal(sigma=1.5).csv','./resources/tokenized_texts_(old).p', categorical=categorical)
    from mix_data import read_mix_data

    texts, valence, arousal = read_mix_data(categorical)

    if option == 'V':
        Y = valence
    elif option == 'A':
        Y = arousal
    else:
        raise Exception('Wrong parameters!')

    lexicon = load_CVAW(extended=using_extended_lexicon)
    d = dict()
    ind = 1 if option == 'V' else 2
    for l in lexicon:
        d[l[0]] = l[ind]

    predicted_ratings = mean_ratings(texts, d, mean_method, Y)
    print(predicted_ratings)
    print(Y)
    out = regression_evaluate(Y, predicted_ratings)

    draw_scatter(Y, predicted_ratings, 'True Values', 'Predicted Values', title='Scatter')

    out2 = cv(predicted_ratings, Y)

    Dims = 'Valence' if option == 'V' else 'Arousal'
    Mean_Method = 'Geometric' if mean_method == 'tf_geo' else 'Arithmetic'
    print('|%s|%s|False|%.3f|%.3f|%.3f|' % (Dims, Mean_Method, out[0], out[1], out[2]))
    print('|%s|%s|True|%.3f|%.3f|%.3f|' % (Dims, Mean_Method, out2[0], out2[1], out2[2]))
