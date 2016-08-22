import json
import os
from my_utils import get_majority_label, pickle_data, unpickle_data
from nltk.tokenize import word_tokenize

_train_path = './snli_1.0/snli_1.0_train.jsonl'
_dev_path = './snli_1.0/snli_1.0_dev.jsonl'
_test_path = './snli_1.0/snli_1.0_test.jsonl'

_pickle_dir = './pickled'


def load_data(file_path, ignore_nonconsenting=True):
    """

    :param file_path: path to file. assumes items are in individual lines as json strings
    :param ignore_nonconsenting: ignores those samples where majority annotaters don't agree
    :return:
    """
    sentence1_list = []
    sentence2_list = []
    labels_list = []
    ignored = 0
    processed = 0
    with open(file_path) as fp:
        for line in fp:
            j = json.loads(line)
            annotater_labels = j['annotator_labels']
            majority_label = get_majority_label(annotater_labels, ignore_nonconsenting)
            if majority_label is None:
                ignored += 1
                continue
            sentence1 = word_tokenize(j['sentence1'])
            if sentence1[-1] == '.': sentence1 = sentence1[:-1]
            sentence2 = word_tokenize(j['sentence2'])
            if sentence2[-1] == '.': sentence2 = sentence2[:-1]
            sentence1_list.append(sentence1)
            sentence2_list.append(sentence2)
            labels_list.append(majority_label)
            processed += 1
            if processed % 10000 == 0: print('processed %d samples' % processed)
    print('ignored %d non-consenting samples' % (ignored))
    return sentence1_list, sentence2_list, labels_list


def print_count(list, listname):
    print('%d items in %s' % (len(list), listname))


def serialize_data(sentence1_list, sentence2_list, labels_list, name):
    if not os.path.exists(_pickle_dir):
        os.makedirs(_pickle_dir)
    fpath = os.path.join(_pickle_dir, name)
    pickle_data([sentence1_list, sentence2_list, labels_list], fpath)


def unserialize_data(name):
    fpath = os.path.join(_pickle_dir, name)
    if not os.path.isfile(fpath):
        print('serialized files does not exist at %s' % fpath)
        raise ValueError
    return unpickle_data(fpath)


if __name__ == '__main__':
    print('loading train data')
    train_s1, train_s2, train_l = load_data(_train_path)
    print_count(train_s1, 'train')
    print('')

    print('loading dev data')
    dev_s1, dev_s2, dev_l = load_data(_dev_path)
    print_count(dev_s1, 'dev')
    print('')

    print('loading test data')
    test_s1, test_s2, test_l = load_data(_test_path)
    print_count(test_s1, 'test')
    print('')

    # serialize_data(train_s1, train_s2, train_l, 'train')
    # serialize_data(dev_s1, dev_s2, dev_l, 'dev')
    # serialize_data(test_s1, test_s2, test_l, 'test')

