#!/usr/bin/env python3
""" Statistical Language Processing (SNLP), Assignment 6
    See <https://snlp2020.github.io/a6/> for detailed instructions.

    Jinghua Xu
"""


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
            # ? if eos:
            label.append(1)
            labels.append(label)

    return utterances, labels


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
    # Exercise 6.2


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
    u, l = read_data('/Users/xujinghua/a6-lahmy98-jinhxu/readdata_test.txt')
    print(u)
    print(l)

    '''
    # Approximate usage of the exercises (not tested).
    u, l = read_data('br-phono.txt')

    train_size = int(0.8 * len(u))
    u_train, l_train = u[:train_size], l[:train_size]
    u_test, l_test = u[train_size:], l[train_size:]

    seg_test = segment(u_train, l_train, u_test)

    evaluate(labels_to_segments(u_test, l_test), seg_test)
    '''
