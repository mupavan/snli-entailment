from reader import unserialize_data
import numpy as np

from my_utils import get_vocab, convert_sentences_to_index_matrix
from custom_layers import HiddenStateLSTM, MaskEatingLambda

from keras.layers import Input, Embedding, Dense, TimeDistributed, Lambda, Flatten, merge
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras.optimizers import Adam

import argparse, sys


UNK = 'unk'
DELIMITER = 'delimiter'
EMPTY = 'empty' # reserved for padded words

label2index = { 'contradiction': 0, 'neutral': 1, 'entailment': 2 }


def get_config():
    parser = argparse.ArgumentParser(description='Entailment with simple attention using LSTMs')
    parser.add_argument('-lstm', action="store", default=128, dest="lstm_dim", type=int)
    parser.add_argument('-epochs', action="store", default=20, dest="nb_epochs", type=int)
    parser.add_argument('-batch', action="store", default=1024, dest="batch_size", type=int)
    parser.add_argument('-emb', action="store", default=50, dest="emb", type=int)
    parser.add_argument('-lr', action="store", default=0.001, dest="lr", type=float)
    parser.add_argument('-verbose', action="store", default=1, dest="verbose", type=int)
    parser.add_argument('-attention', action="store", default=False, dest="use_attention", type=bool)
    opts = parser.parse_args(sys.argv[1:])
    print "lstm_dim", opts.lstm_dim
    print "nb_epochs", opts.nb_epochs
    print "batch_size", opts.batch_size
    print "learning_rate", opts.lr
    print "emb", opts.emb
    print "using attention", opts.use_attention
    return opts


def build_word2index(vocab):
    word2index = {}
    word2index[EMPTY] = 0
    word2index[DELIMITER] = 1
    word2index[UNK] = 2
    for word in vocab:
        word2index[word] = len(word2index)
    return word2index


def build_lstm_model(s1_len, s2_len, nb_words, nb_labels, word_dim, lstm_dim, lr):
    s1_input = Input(shape=(s1_len,), dtype='int32')
    s2_input = Input(shape=(s2_len,), dtype='int32')

    s1_embedding = Embedding(input_dim=nb_words, output_dim=word_dim, input_length=s1_len, mask_zero=True)(s1_input)
    s2_embedding = Embedding(input_dim=nb_words, output_dim=word_dim, input_length=s2_len, mask_zero=True)(s2_input)

    # returns output, *hidden_states
    temp = HiddenStateLSTM(output_dim=lstm_dim, dropout_W=0., return_sequences=False)(s1_embedding)
    premise_enc = temp[0]
    premise_states = temp[1:]

    temp = HiddenStateLSTM(output_dim=lstm_dim, dropout_W=0., return_sequences=False)([s2_embedding] + premise_states)
    hypothesis_cond_enc = temp[0]

    # removing mask
    hypothesis_cond_enc = MaskEatingLambda(lambda x, mask: x)(hypothesis_cond_enc)

    predictions = Dense(nb_labels, activation='softmax')(hypothesis_cond_enc)
    model = Model(input=[s1_input, s2_input], output=predictions)
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def build_attention_lstm_model(s1_len, s2_len, nb_words, nb_labels, word_dim, lstm_dim, lr):
    s1_input = Input(shape=(s1_len,), dtype='int32')
    s2_input = Input(shape=(s2_len,), dtype='int32')

    s1_embedding = Embedding(input_dim=nb_words, output_dim=word_dim, input_length=s1_len, mask_zero=True)(s1_input)
    s2_embedding = Embedding(input_dim=nb_words, output_dim=word_dim, input_length=s2_len, mask_zero=True)(s2_input)

    # returns output, *hidden_states
    temp = HiddenStateLSTM(output_dim=lstm_dim, dropout_W=0., return_sequences=True)(s1_embedding)
    premise_outputs = temp[0]
    premise_states = temp[1:]

    temp = HiddenStateLSTM(output_dim=lstm_dim, dropout_W=0., return_sequences=False)([s2_embedding] + premise_states)
    hypothesis_cond_enc = temp[0]
    # removing mask
    hypothesis_cond_enc = MaskEatingLambda(lambda x, mask: x)(hypothesis_cond_enc)

    # machinery for attention
    M_enc = TimeDistributed(Dense(lstm_dim, activation='linear'))(premise_outputs)
    M_hn = Dense(lstm_dim, activation='linear')(hypothesis_cond_enc)
    M = merge(
        [M_enc, M_hn],
        mode=lambda x: x[0] * K.permute_dimensions(x[1], (0, 'x', 1)),
        output_shape=lambda shapes: shapes[0])

    premise_mask = MaskEatingLambda(lambda x, mask: mask, output_shape=lambda shape: (shape[0], shape[1]))(s1_embedding)
    # attentions
    alpha = TimeDistributed(Dense(1, activation='linear'))(M)
    alpha = Flatten()(alpha)
    alpha = Lambda(lambda x: K.exp(x))(alpha)
    alpha = merge(
        [alpha, premise_mask],
        mode=lambda inputs: inputs[0] * inputs[1],
        output_shape=lambda shapes: shapes[0])
    alpha = Lambda(lambda x: x / K.sum(x, axis=1, keepdims=True))(alpha)

    r = merge(
        [M_enc, alpha],
        mode=lambda x: K.sum(x[0] * K.permute_dimensions(x[1], (0, 1, 'x')), axis=1),
        output_shape=lambda shapes: (shapes[0][0], shapes[0][2])
    )

    h_star = merge([r, hypothesis_cond_enc], mode='concat', concat_axis=1)
    h_star = Dense(lstm_dim, activation='tanh')(h_star)

    predictions = Dense(nb_labels, activation='softmax')(h_star)
    model = Model(input=[s1_input, s2_input], output=predictions)
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train_model(model, x_s1, x_s2, x_l, batch_size=1024):
    model.fit([x_s1, x_s2], to_categorical(x_l), batch_size=batch_size, verbose=1)


def get_indexed_data(s1, s2, labels, word2index=None, maxlen_s1=None, maxlen_s2=None):
    # s1, s2, labels = unserialize_data('dev')
    x_l = np.array( [ label2index[l] for l in labels ] )
    if word2index is None:
        vocab = get_vocab(s1)
        vocab = get_vocab(s2, vocab)
        word2index = build_word2index(vocab)
    x_s1 = convert_sentences_to_index_matrix( s1, word2index, UNK, maxlen_s1 )
    x_s2 = convert_sentences_to_index_matrix( s2, word2index, UNK, maxlen_s2 )
    return x_s1, x_s2, x_l, word2index


if __name__ == '__main__':
    config = get_config()

    print('loading data')
    train_s1, train_s2, train_labels = unserialize_data('train')
    dev_s1, dev_s2, dev_labels = unserialize_data('dev')
    test_s1, test_s2, test_labels = unserialize_data('test')

    print('convering data to numpy arrays')
    train_x_s1, train_x_s2, train_x_l, word2index = get_indexed_data(train_s1, train_s2, train_labels)
    maxlen_s1 = train_x_s1.shape[1]
    maxlen_s2 = train_x_s2.shape[1]

    dev_x_s1, dev_x_s2, dev_x_l, _ = get_indexed_data(dev_s1, dev_s2, dev_labels, word2index, maxlen_s1, maxlen_s2)
    test_x_s1, test_x_s2, test_x_l, _ = get_indexed_data(test_s1, test_s2, test_labels, word2index, maxlen_s1, maxlen_s2)

    print('building the model')
    if config.use_attention:
        model = build_attention_lstm_model(maxlen_s1, maxlen_s2, len(word2index), len(label2index), config.emb, config.lstm_dim, config.lr)
    else:
        model = build_lstm_model(maxlen_s1, maxlen_s2, len(word2index), len(label2index), config.emb, config.lstm_dim, config.lr)
    print('fit')
    model.fit(
        [train_x_s1, train_x_s2],
        to_categorical(train_x_l),
        validation_data=([dev_x_s1, dev_x_s2], to_categorical(dev_x_l)),
        nb_epoch=config.nb_epochs,
        batch_size=config.batch_size,
        verbose=config.verbose,
    )
    test_y_hat = np.argmax(model.predict([test_x_s1, test_x_s2]), axis=1)
    test_y_act = test_x_l
