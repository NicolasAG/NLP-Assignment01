"""
Created on Oct 10, 2016

@author: Nicolas A.G.
"""

import os
import argparse

from nltk.tag import hmm
from nltk.probability import FreqDist, LaplaceProbDist, MLEProbDist


ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
            'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', ',', '.']


################################################################################
# reading data set

def normalize_sentence(sentence, strip=True):
    """
    Normalize a given sentence to the restricted alphabet we have.
    :param sentence: the string to normalize
    :param strip: whether to remove beginning and ending spaces.
    :return: a new string with the following features:
    - lowercase characters only
    - characters only taken from the restricted ALPHABET we defined
    - (if strip=True) no extra space at the beginning and the end of the sentence
    """
    if strip:
        return ''.join([new_char for new_char in sentence.lower().strip() if new_char in ALPHABET])
    else:
        return ''.join([new_char for new_char in sentence.lower() if new_char in ALPHABET])


def read_cipher(cipher_folder):
    """
    Read data set and return [X_train, Y_train, X_test, and Y_test]
     where each of these are list of strings for each sentence respectively
     coming from train_cipher.txt, train_plain.txt, test_cipher.txt, test_plain.txt
    """

    data = [[], [], [], []]  # X_train, Y_train, X_test, Y_test

    file_names = ['train_cipher.txt', 'train_plain.txt', 'test_cipher.txt', 'test_plain.txt']

    for (file_name, i) in zip(file_names, range(4)):
        text_file = open(os.path.join(cipher_folder, file_name))
        for line in text_file:
            # NEVER STRIP THE TRAINING DATA NOR THE CIPHERED TEXT!
            # just make sure it's all lowercase, and remove \r\n from the end of each sentence.
            data[i].append(normalize_sentence(line, strip=False))

    return data


################################################################################
# returning sequence of labels
def get_sequences(ciphered_text, plain_text):
    """
    Make a list of labeled sequences for each sentence in X and Y.
    :param ciphered_text: array of strings representing 1 line of observable items.
    :param plain_text: array of strings representing 1 line of hidden tags.
    :return: list of the form (sequence_for_sentence_1, ..., sequence_for_sentence_n) where each
            sequence is a list of 2-tuple (item, tag) for each character in that specific sentence.
    """
    sequences = []

    for sentence_o, sentence_h in zip(ciphered_text, plain_text):
        # for each sentence in X and Y:
        #  append a list of tuple ('char i in sentence_o', 'char i in sentence_h')
        #  where 1 tuple = 1 character.
        # list("sentence") = ['s',...,'e']
        sequences.append(zip(list(sentence_o), list(sentence_h)))

    return sequences


def main():
    parser = argparse.ArgumentParser(description="Decipher some text using HMM.")
    parser.add_argument("cipher_folder",
                        help="path to the folder containing the cipher and plain text files")
    parser.add_argument("-laplace",
                        action="store_true",
                        help="turn on laplace smoothing")
    parser.add_argument("-lm",
                        action="store_true",
                        help="turn on the language model from assignment1")
    args = parser.parse_args()

    [x_train, y_train, x_test, y_test] = read_cipher(args.cipher_folder)
    # each item is of the form: ["sentence 1", "sentence 2", ..., "sentence n"]
    # print "x_train\n", x_train
    # print "y_train\n", y_train
    # print "x_test\n", x_test
    # print "y_test\n", y_test

    print "laplace:", args.laplace
    print "lm:", args.lm
    print "- - -"

    ##
    # Training
    ##
    labeled_sequences = get_sequences(x_train, y_train)  # of the form [ [('#','#')...], [...], ...]

    # Get the Frequency Distribution for the language model from assignment 1
    lm_fd = None
    if args.lm:
        # TODO: create a FreqDist for the language model
        # it can be saved, check here: http://www.nltk.org/howto/probability.html
        pass

    trainer = hmm.HiddenMarkovModelTrainer()

    if args.laplace:
        laplace = lambda fd, bins: LaplaceProbDist(fd+lm_fd if lm_fd else fd, bins)
        tagger = trainer.train_supervised(labeled_sequences, estimator=laplace)  # estimator = Laplace Smoothing
    else:
        mle = lambda fd, bins: MLEProbDist(fd+lm_fd if lm_fd else fd)
        tagger = trainer.train_supervised(labeled_sequences, estimator=mle)  # estimator = MLE

    ##
    # Prediction
    ##
    for i in range(len(x_test)):  # for each sentence in test set
        print "cipher:\t\t", x_test[i]
        print "plain:\t\t", y_test[i]
        # predict the real characters based on the list of ciphered characters:
        predictions = tagger.best_path(list(x_test[i]))  # get array of predicted characters
        print "prediction:\t", ''.join(predictions)
        print "- - -"

    test_sequences = get_sequences(x_test, y_test)  # of the form [ [('#','#')...], [...], ...]
    tagger.test(test_sequences)


if __name__ == '__main__':
    main()
