import numpy as np

from mix_data import read_mix_data
from load_data import load_CVAW
from evaluate import regression_evaluate

def predict(sentence, lexicon, method):
    words = sentence.split()
    sentiment_words = sorted(list(set(words) & set(lexicon.keys())))
    print('sentiment words: %s' % str(sentiment_words))
    sentiment_values = [lexicon[i] for i in sentiment_words]
    print('correpsonding sentiment values: %s' % sentiment_values)

    if len(sentiment_words)==0:
        print('No sentiment words found in this sentence!')
        return 5
    else:
        if method == 'TF_mean':
            return np.sum(sentiment_values)/len(sentiment_words)
        elif method == 'Geo_mean':
            return np.prod(sentiment_values)**(1/len(sentiment_words))
        else:
            raise Exception('Wrong values')


def va_prediction(sentences, lexicon, true_values):
    arithmetic, geometric =[], []
    for sentence in sentences:
        print(sentence)
        predicted_value_a = predict(sentence, lexicon, 'TF_mean')
        arithmetic.append(predicted_value_a)
        print('The predicted values is (using Arithmetic Average): %s'% predicted_value_a)
        predicted_value_g = predict(sentence, lexicon, 'Geo_mean')
        print('The predicted values is (using Geometric  Average): %s'% predicted_value_g)
        geometric.append(predicted_value_g)
    return arithmetic, geometric

if __name__ == '__main__':
    ########################################### Hyper-parameters ###########################################
    target = 'valence' # values: "valence", "arousal"
    categorical = 'political'  # values: 'all', "book", "car", "laptop", "hotel", "news", "political"
    ########################################################################################################
    texts, valence, arousal = read_mix_data(categorical)

    lexicon = load_CVAW()
    d = dict()
    if target == 'valence':
        ind = 1
        true_values = valence
        print('Valence prediction...')
    elif target == 'arousal':
        ind = 2
        true_values = arousal
        print('Arousal preddiction...')
    for l in lexicon:
        d[l[0]] = l[ind]

    arithmetic, geometric = va_prediction(texts, d, true_values)
    print('Prediction result (arithmetic average):')
    regression_evaluate(true_values, arithmetic)
    print('Prediction result (geometric average):')
    regression_evaluate(true_values, geometric)