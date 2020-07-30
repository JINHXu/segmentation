#!/usr/bin/env python3
""" Statistical Language Processing (SNLP), Assignment 6
    See <https://snlp2020.github.io/a6/> for detailed instructions.

    Jinghua Xu
"""
import numpy as np
from sklearn import preprocessing

import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM



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
    encoded_u_train = we.transform(u_train, flat=False)
    '''
    print(encoded_u_train)
    print(encoded_u_train.shape)
    print(we._char2idx)

    print(l_train)
    print(type(l_train))
    '''


    # pad labels
    for l in l_train:
        if len(l) < encoded_u_train.shape[1]:
            l.extend([2]*(encoded_u_train.shape[1]-len(l)))
    
    l_train = np.array(l_train)

    '''
    print(len(l_train[0]))
    print(len(l_train[1]))
    print(len(l_train[2]))
    '''

    output_dim = l_train.shape[1]

    # train
    grnn = Sequential()
    grnn.add(LSTM(64, input_shape=(
        encoded_u_train.shape[1], encoded_u_train.shape[2]), activation='relu'))
    grnn.add(Dense(output_dim, activation='softmax'))

    grnn.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    grnn.fit(encoded_u_train, l_train)

    # predict

    # encode test utterance
    encoded_u_test = we.transform(u_test, flat = False)

    l_test_pred = grnn.predict(encoded_u_test)
    print(l_test_pred)

    # get rid of the tailling padded numbers?

    # labels to segments
    pred_seg = labels_to_segments(u_test, l_test_pred)
    print(pred_seg)


def evaluate(gold_seg, pred_seg):
    """ Calculate and print out boundary/word/lexicon precision recall and F1 scores.

    See the assignment sheet for definitions of the metrics.

    Parameters:
    -----------
    gold_seg:  A list of sequences of gold-standard words 
    pred_seg:  Predicted segments.

    Returns: None
    """
    # Exercise 6.3


if __name__ == '__main__':

    

    # test read_data
    u, l = read_data('/Users/xujinghua/a6-lahmy98-jinhxu/readdata_test.txt'  # , eos=''
                     )
    print(u)
    print(l)

    segment(u, l, u)

    '''

    # test labels_to_segments
    print(labels_to_segments(u, l))
    



    
    # Approximate usage of the exercises (not tested).
    u, l = read_data('br-phono.txt')





    

    # train-test split
    train_size = int(0.8 * len(u))
    u_train, l_train = u[:train_size], l[:train_size]
    u_test, l_test = u[train_size:], l[train_size:]

    # train a gated RNN for predicting the boundaries(1/0 tags)
    seg_test = segment(u_train, l_train, u_test)

    evaluate(labels_to_segments(u_test, l_test), seg_test)
'''
