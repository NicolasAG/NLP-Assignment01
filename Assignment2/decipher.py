"""
Created on Oct 10, 2016

@author: Nicolas A.G.
"""

import os
import argparse

from nltk.tag import hmm


################################################################################
# reading data set
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
            data[i].append(line)

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
    # print "x_train"; print x_train
    # print "y_train"; print y_train
    # print "x_test"; print x_test
    # print "y_test"; print y_test

    print "laplace:", args.laplace
    print "lm:", args.lm

    ##
    # Training
    ##
    labeled_sequences = get_sequences(x_train, y_train)  # of the form [ [('#','#')...], [...], ...]

    trainer = hmm.HiddenMarkovModelTrainer()
    tagger = trainer.train_supervised(labeled_sequences)  # estimator = None -> MLE is default

    ##
    # Prediction
    ##
    for i in range(len(x_test)):  # for each sentence in test set
        print "cipher:\n", x_test[i]
        print "plain:\n", y_test[i]
        # predict the real characters based on the list of ciphered characters:
        predictions = tagger.best_path(list(x_test[i]))  # get array of predicted characters
        print "prediction:\n", ''.join(predictions)
        print "- - -"

    test_sequences = get_sequences(x_test, y_test)  # of the form [ [('#','#')...], [...], ...]
    tagger.test(test_sequences)


if __name__ == '__main__':
    main()
