# [Segmentation as Sequence labeling](https://snlp2020.github.io/a6/)


Segmentation with Gated RNN, as a sequence, label each character as 'beginning of a word' (B, or 1)
or 'inside a word' (I, or 0). Assume that there will be no 'other' (O) tag
(for example space characters for tokenization,
or words that do not belong to any named entity).


In particular the data comes from the
[CHILDES](https://childes.talkbank.org/)
(originally collected by [Bernstein-Ratner
(1987)](https://books.google.de/books?id=fBapP38RhwsC&lpg=PA159&ots=48MF3EUK6w&dq=The%20phonology%20of%20parent-child%20speech&lr&pg=PA159#v=onepage&q&f=false)
 processed by [Brent & Cartwright (1996)](https://www.sciencedirect.com/science/article/pii/S0010027796007196)
 into to present form).
The data comes as [phonemic transcriptions](br-phono.txt)
and [regular text](br-text.txt).
In child language acquisition research,
the phonemic transcriptions are used.

Each line in the data files contains an utterance,
and the word boundaries are indicated with space characters.


Library: [Keras](https://www.tensorflow.org/api_docs/python/tf/keras)

### 6.1 Reading the data

the function `read_data()` reads a data file formatted as described,
and returns utterances without spaces,
as well as `0` and `1` labels corresponding to 'inside a word'
and 'beginning of a word' respectively.
Also include a special 'end-of-utterance' symbol (`#`)
which is treated as a word.

### 6.2 RNN sequence labeller

the function `segment()` trains a (gated) RNN for predicting the boundaries
(B/I or 1/0 tags) using the unsegmented
input returned by `read_data()` described above.
The function should return sequences of segmented test utterances.


### 6.3 Evaluating the output

the function `evaluate()` takes 
gold-standard segmentations and predicted segmentations,
and produces the following metrics:

- _Boundary_ precision (BP), recall (BR) and F1 score: these scores should
    indicate the ratio of the predicted boundaries that are correct
    according to the gold standard (precision),
    the ratio of the gold-standard boundaries
    that were correctly predicted (recall),
    and their harmonic mean (F1 score).
- _Word_ precision (WP), recall (WR) and F1 score: similar to above, but to
    count a true positive, both boundaries of a word should be
    identified correctly.
- _Lexicon_ precision (LP), recall (LR) and F1 score:
    similar to word precision, but each word type (unique word)
    counts only once.
