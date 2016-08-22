from collections import Counter
from operator import itemgetter
import cPickle as pickle
from keras.preprocessing.sequence import pad_sequences


def get_majority_label(annotater_labels, ignore_nonconsenting=False):
    n = len(annotater_labels)
    c = Counter(annotater_labels)
    max_label, max_count = max(c.iteritems(), key=itemgetter(1))
    if ignore_nonconsenting and max_count < float(n) / 2:
        return None
    else:
        return max_label


def pickle_data(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle_data(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def get_vocab(sentences, vocab=None):
    vocab = set() if vocab is None else vocab
    for words in sentences:
        for word in words:
            vocab.add(word)
    return vocab


def convert_sentences_to_index_matrix(sentences, word2index, unk, maxlen=None):
    def convert_word_to_index(word):
        if word in word2index:
            return word2index[word]
        else:
            return word2index[unk]
    indexed_sentences = []
    for words in sentences:
        indexed_sentences.append(map(convert_word_to_index, words))
    if maxlen is None:
        return pad_sequences(indexed_sentences, dtype='int32', padding='post', value=0)
    else:
        return pad_sequences(indexed_sentences, maxlen=maxlen, dtype='int32', padding='post', value=0)
