#!/usr/bin/env python3
""" Statistical Language Processing (SNLP), Assignment 6
    See <https://snlp2020.github.io/a6/> for detailed instructions.
    Course:      Statistical Language processing - SS2020
    Assignment:  a6
    Author(s):   Lisa Seneberg, Jinghua Xu
    Description: segementation with gated rnn
 
 Honor Code:  I pledge that this program represents my own work.
"""
import numpy as np
from sklearn import preprocessing

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Masking

from sklearn.metrics import f1_score, precision_score, recall_score


def read_data(filename, eos='#'):
    """ Read an input file

    Parameters:
    -----------
    filename:  The input file
    eos:       Symbol to add to the end of each sequence (utterance)

    Returns: (a tuple)
    -----------
    utterances: A list of strings without white spaces 
    labels:     List of sequences  of 0/1 labels. '1' indicates that the
                corresponding character in 'utterances' begins a word (B),
                '0' indicates that it is inside a word (I).
    """
    utterances = []
    labels = []

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # utterance
            utterance = line.replace(" ", "") + eos
            utterances.append(utterance)

            # label
            label = []
            for word in line.split(' '):
                # beginning of a word
                label.append(1)
                # in side of a word
                inw = [0] * (len(word)-1)
                label.extend(inw)
            # end of utterance
            if eos:
                label.append(1)
            labels.append(label)

    return utterances, labels


class WordEncoder:
    """An encoder for a sequence of words.
    The encoder encodes each word as a sequence of one-hot characters.
    The words that are longer than 'maxlen' is truncated, and
    the words that are shorter are padded with 0-vectors.
    Two special symbols, <s> and </s> (beginning of sequence and
    end of sequence) should be prepended and appended to every word
    before padding or truncation. You should also reserve a symbol for
    unknown characters (distinct from the padding).
    The result is either a 2D vector, where all character vectors
    (including padding) of a word are concatenated as a single vector,
    o a 3D vector with a separate vector for each character (see
    the description of 'transform' below and the assignment sheet
    for more detail.
    Parameters:
    -----------
    maxlen:  The length that each word (including <s> and </s>) is padded
             or truncated. If not specified, the fit() method should
             set it to cover the longest word in the training set. 
    """

    def __init__(self, maxlen=None):
        # to be set up in fit()
        self._maxlen = maxlen
        self._char2idx = dict()
        self._nchars = len(self._char2idx)

    def fit(self, words):
        """Fit the encoder using words.
        All collection of information/statistics from the training set
        should be done here.
        Parameters:
        -----------
        words:  The input words used for training.
        Returns: None
        """
        setUPmaxlen = False
        if self._maxlen is None:
            self._maxlen = 0
            setUPmaxlen = True

        # special symbols
        self._char2idx['<s>'] = 0
        self._char2idx['</s>'] = 1
        # reserve for unknown chararacters
        self._char2idx['uk'] = 2

        # current index
        idx = 3
        # chars in words
        for word in words:
            if len(word) > self._maxlen and setUPmaxlen:
                self._maxlen = len(word)
            for char in word:
                if char not in self._char2idx:
                    self._char2idx[char] = idx
                    idx += 1
        self._nchars = len(self._char2idx)

    def transform(self, words, pad='right', flat=True):
        """ Transform a sequence of words to a sequence of one-hot vectors.
        Transform each character in each word to its one-hot representation,
        combine them into a larger sequence, and return.
        The returned sequences formatted as a numpy array with shape 
        (n_words, max_wordlen * n_chars) if argument 'flat' is true,
        (n_words, max_wordlen, n_chars) otherwise. In both cases
        n_words is the number of words, max_wordlen is the maximum
        word length either set in the constructor, or determined in
        fit(), and n_chars is the number of unique characters.
        Parameters:
        -----------
        words:  The input words to encode
        pad:    Padding direction, either 'right' or 'left'
        flat:   Whether to return a 3D array or a 2D array (see above
                for explanation)
        Returns: (a tuple)
        -----------
        encoded_data:  encoding the input words (a 2D or 3D numpy array)
        """

        # params check
        if isinstance(flat, bool) and (pad == 'right' or pad == 'left'):
            pass
        else:
            raise ValueError(
                "Illegal Argument! pad can only be 'right' or 'left', flat has to be bool!")
        encoded_words = []
        for word in words:
            word = list(word)
            encoded_word = []
            # prepend special char
            word.insert(0, '<s>')
            # append special char
            word.append('</s>')
            if len(word) > self._maxlen:
                # truncation
                word = word[:self._maxlen]
            for char in word:
                char_vec = [0]*self._nchars
                if char in self._char2idx:
                    idx = self._char2idx[char]
                    char_vec[idx] = 1
                else:
                    # unknown char
                    char = 'uk'
                    idx = self._char2idx[char]
                    char_vec[idx] = 1
                if flat:
                    encoded_word = encoded_word + char_vec
                else:
                    encoded_word.append(char_vec)
            if len(word) < self._maxlen:
                # padding
                padding = [0]*self._nchars
                if pad == 'right':
                    for _ in range(self._maxlen-len(word)):
                        if flat:
                            encoded_word = encoded_word + padding
                        else:
                            encoded_word.append(padding)
                else:
                    for _ in range(self._maxlen-len(word)):
                        if flat:
                            encoded_word = padding + encoded_word
                        else:
                            encoded_word.insert(0, padding)
            encoded_words.append(encoded_word)
        return np.array(encoded_words)


def labels_to_segments(utterances, labels):
    """Return segmented utterances corresponding (B/I) labels,

    Given a sequence of utterances like
        ["nightnight#", "daddy#", "akitty#"]
    and corresponding lables like,
        [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], 
         [1, 0, 0, 0, 0, 1], 
         [1, 1, 0, 0, 0, 0, 1]]
    return the segmented utterances similar to:
        [["night", "night", "#"], ["daddy", "#"], ["a", "kitty", "#"]]

    Note that to use it with model predictions, you should assume a
    number larger than 0.5 in the 'labels' array indicates a boundary.

    Parameters:
    -----------
    utterances:  The list of unsegmented utterances.
    labels:      The B/O labels (probabilities) for given utterances.

    Returns: 
    segments:  Segmented utterances.

    """
    # You will need this function to avoid duplicating code for
    # the main part that re-segments the test utterances,
    # and in the segment() function below.

    segments = []
    for utterance, label in zip(utterances, labels):
        # this solution requires an eos
        eos = True
        if label[-1] == 0:
            eos = False
            label.append(1)

        segment = []
        # get the indices for slicing utterance
        indices = [0]
        for i in range(1, len(label)):
            if label[i] == 1:
                start_i = indices.pop()
                segment.append(utterance[start_i:i])
                indices.append(i)

        # eos
        if eos:
            segment.append(utterance[-1])

        segments.append(segment)
    return segments


def segment(u_train, l_train, u_test):
    """ Train an RNN sequence labeller on the training set, return segmented test set.
    Parameters:
    -----------
    u_train:  The list of unsegmented utterances for training
    l_train:  Training set labels, corresponding to each character in 'u_train'
    u_test:   Unsegmented test input

    The format of the u_train and u_test is similar to 'utterances'
    returned by 'read_data()'.


    Returns: 
    pred_seg:  Predicted segments, a list of list of strings, each
               corresponding to a predicted word.
    """
    # encode words
    we = WordEncoder()
    we.fit(u_train)
    encoded_u_train = we.transform(u_train, pad='left', flat=False)

    # pad labels
    padded_l_train = keras.preprocessing.sequence.pad_sequences(l_train)

    _, timesteps, featurelen = encoded_u_train.shape
    output_dim = padded_l_train.shape[1]

    # train
    grnn = Sequential()

    grnn.add(Masking(input_shape=(timesteps, featurelen)))
    # a conventional choice of number of hidden units
    grnn.add(LSTM(64))
    grnn.add(Dense(output_dim, activation='sigmoid'))

    grnn.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

    grnn.summary()

    grnn.fit(encoded_u_train, padded_l_train)

    # predict
    encoded_u_test = we.transform(u_test, flat=False)
    l_test_pred = grnn.predict(encoded_u_test)

    # process output
    bi_opt = []
    for seq, label in zip(u_test, l_test_pred):
        label = label[-len(seq):]
        tmp = []
        for l in label:
            if l >= 0.5:
                tmp.append(1)
            else:
                tmp.append(0)
        bi_opt.append(tmp)

    # labels to segments
    pred_seg = labels_to_segments(u_test, bi_opt)
    
    return pred_seg


def evaluate(gold_seg, pred_seg):
    """ Calculate and print out boundary/word/lexicon precision recall and F1 scores.

    See the assignment sheet for definitions of the metrics.

    Parameters:
    -----------
    gold_seg:  A list of sequences of gold-standard words 
    pred_seg:  Predicted segments.

    Returns: None
    """
    # Our treatment of 'end-of-sequence' symbol(a boundary before the first word or after # should not add to true positives) determines that 'eos' has alway to be assumed non-empty
    eos = gold_seg[0][-1]

    # get the labels in a flat list
    gold_l = []
    pred_l = []
    for segs in gold_seg:
        l = []
        for seg in segs:
            l.append(1)
            inw = [0] * (len(seg)-1)
            l.extend(inw)
        # credit no boundaries before the first word or after #
        l.pop(0)
        gold_l.extend(l)
    for segs in pred_seg:
        l = []
        for seg in segs:
            l.append(1)
            inw = [0] * (len(seg)-1)
            l.extend(inw)
        # credit no boundaries before the first word or after #
        l.pop(0)
        pred_l.extend(l)

    # boundary scores
    bp = precision_score(gold_l, pred_l)
    br = recall_score(gold_l, pred_l)
    bf1 = f1_score(gold_l, pred_l)

    # words TP + FP(word precision denominator): number of words in pred_seg disregard of the 'end-of-sequence' symbol
    word_tpnfp = 0
    for s in pred_seg:
        word_tpnfp += len(s)
        if s[-1] == eos:
            word_tpnfp -= 1

    # words TP + FN(word recall denominator): number of words in gold_seg disregard of the 'end-of-sequence' symbol
    word_tpnfn = 0
    for s in gold_seg:
        word_tpnfn += len(s)-1

    # words TP
    word_tp = 0
    # get a list of boundaries
    pred_b = []
    left_b = 0
    right_b = 0
    for s in pred_seg:
        for w in s:
            # it is assumed the data is free of 'eos'
            if w == eos:
                # 'eos' does not count towards TP
                continue
            right_b = left_b + len(w) - 1
            pred_b.append((left_b, right_b))
            left_b = right_b + 1

    gold_b = []
    left_b = 0
    right_b = 0
    for s in gold_seg:
        for w in s:
            # it is assumed the data is free of 'eos'
            if w == eos:
                # 'eos' does not count towards TP
                continue
            right_b = left_b + len(w) - 1
            gold_b.append((left_b, right_b))
            left_b = right_b + 1

    for bd in gold_b:
        if bd in pred_b:
            word_tp += 1

    wp = word_tp/word_tpnfp
    wr = word_tp/word_tpnfn
    wf1 = (2*wp*wr)/(wp+wr)

    # get lexicon
    gold_lexicon = set()
    pred_lexicon = set()

    for s in gold_seg:
        for w in s:
            if w == eos:
                continue
            gold_lexicon.add(w)

    for s in pred_seg:
        for w in s:
            if w == eos:
                continue
            pred_lexicon.add(w)

    lexicon_tp = len(gold_lexicon & pred_lexicon)
    lexicon_fp = len(pred_lexicon-gold_lexicon)
    lexicon_fn = len(gold_lexicon-pred_lexicon)

    lp = lexicon_tp/(lexicon_tp+lexicon_fp)
    lr = lexicon_tp/(lexicon_tp+lexicon_fn)
    lf1 = (2*lp*lr)/(lp+lr)

    # print statistics
    print('Evaluation:\n')

    print(f'Boundary precision: {bp}')
    print(f'Boundary recall: {br}')
    print(f'Boundary F1: {bf1}\n')

    print(f'Word precision: {wp}')
    print(f'Word recall: {wr}')
    print(f'Word F1: {wf1}\n')

    print(f'Lexicon precision: {lp}')
    print(f'Lexicon recall: {lr}')
    print(f'Lexicon F1: {lf1}\n')


if __name__ == '__main__':
    # Approximate usage of the exercises (not tested).
    u, l = read_data('br-phono.txt')

    train_size = int(0.8 * len(u))
    u_train, l_train = u[:train_size], l[:train_size]
    u_test, l_test = u[train_size:], l[train_size:]

    seg_test = segment(u_train, l_train, u_test)

    evaluate(labels_to_segments(u_test, l_test), seg_test)
