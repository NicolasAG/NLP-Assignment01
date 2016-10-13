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
        file = open(os.path.join(cipher_folder, file_name))
        for line in file:
            data[i].append(line)

    return data

################################################################################
# returning sequence of labels
def get_sequences(X, Y):
    """
    Make a list of labeled sequences for each sentence in X and Y.
    :param X: array of strings representing 1 line of observable items.
    :param Y: array of strings representing 1 line of hidden tags.
    :return: list of the form (sequence_for_sentence_1, ..., sequence_for_sentence_n) where each
            sequence is a list of 2-tuple (item, tag) for each character in that specific sentence.
    """
    sequences = []

    for sentence_o, sentence_h in zip(X, Y):
        # for each sentence in X and Y,
        #  list("sentence") = ['s',...,'e']
        # append a list of tuple ('char i in sentence_o', 'char i in sentence_h')
        #  where 1 tuple = 1 character.
        sequences.append(zip(list(sentence_o), list(sentence_h)))

    return sequences


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


[X_train, Y_train, X_test, Y_test] = read_cipher(args.cipher_folder)
# each item is of the form: ["sentence 1", "sentence 2", ..., "sentence n"]
#print "X_train"; print X_train
#print "Y_train"; print Y_train
#print "X_test"; print X_test
#print "Y_test"; print Y_test

print "laplace:", args.laplace
print "lm:", args.lm

labeled_sequences = get_sequences(X_train, Y_train)  # of the form [ [('#','#')...], [...], ...]

trainer = hmm.HiddenMarkovModelTrainer()
tagger = trainer.train_supervised(labeled_sequences)  # estimator = None -> MLE is default

for i in range(len(X_test)):  # for each sentence in test set
    print "cipher:"; print X_test[i]
    print "plain:"; print Y_test[i]
    # predict the real characters based on the list of ciphered characters:
    predictions = tagger.best_path(list(X_test[i]))  # get array of predicted characters
    print "prediction:"; print ''.join(predictions)
    print "- - -"

test_sequences = get_sequences(X_test, Y_test)  # of the form [ [('#','#')...], [...], ...]
tagger.test(test_sequences)

