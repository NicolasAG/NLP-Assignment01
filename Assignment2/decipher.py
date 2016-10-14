"""
Created on Oct 10, 2016

@author: Nicolas A.G.
"""

import os
import argparse
import codecs

from nltk.tag import hmm
from nltk.probability import LaplaceProbDist, MLEProbDist, ConditionalFreqDist, ConditionalProbDist
from nltk.tokenize import sent_tokenize


ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
            'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', ',', '.']


################################################################################
# reading data set
def normalize_sentence(sentence):
    """
    Normalize a given sentence to the restricted alphabet we have.
    If a character in the sentence is not in the ALPHABET, we simply ignore him, no replacement done.
    :param sentence: the string to normalize
    :return: a new string with the following features:
    - lowercase characters only
    - characters only taken from the restricted ALPHABET we defined
    """
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
            # just making sure it's all lowercase, and remove \r\n from the end of each sentence.
            data[i].append(normalize_sentence(line))

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


################################################################################
# reading data set from the first Assignment
def read_tac():
    """
    Read data set from first assignment and returns array of sentences
    """
    sentences = []
    for year in ['2010', '2011']:
        sub_folder = '../data/tac%s' % year

        if year == '2010':
            template = 'tac10-%04d.txt'
            start, end = 1, 921
        elif year == '2011':
            template = 'tac11-%04d.txt'
            start, end = 921, 1801
        else:
            print "no data for the year ", year
            return

        for i in xrange(start, end):
            text_file = os.path.join(sub_folder, template % i)
            text = codecs.open(text_file, 'r', encoding='utf-8').read()
            text = codecs.encode(text, 'ascii', 'ignore')

            file_sentences = sent_tokenize(text)  # takes care of stripping leading and tailing spaces
            for file_sentence in file_sentences:
                # normalize the sentence according to our alphabet
                sentences.append(normalize_sentence(file_sentence))

    return sentences


def get_cond_samples(sentences):
    """
    Return a list of (condition, sample) tuples from a list of sentences.
    Because we want transition probabilities, the condition is the current character,
    and the sample is the next character in the sentence (minus the last one which is
    usually '.')
    :param sentences: array of string sentences
    :return: a list of (condition, sample) tuples.
    """
    cond_samples = []

    for sentence in sentences:
        # for each sentence:
        #  append a list of tuple ('char i', 'char i+1')
        # list("sentence"[:-1]) = ['s','e','n','t','e','n','c'] <-- remove last char = conditions
        # list("sentence"[1:])  = ['e','n','t','e','n','c','e'] <-- remove first char = samples
        cond_samples.extend(zip(list(sentence[:-1]), list(sentence[1:])))

    return cond_samples


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

    print "laplace:", args.laplace
    print "lm:", args.lm

    ##
    # Training
    ##
    trainer = hmm.HiddenMarkovModelTrainer()

    labeled_sequences = get_sequences(x_train, y_train)  # of the form [ [('#','#')...], [...], ...]

    if args.laplace:
        estimator = lambda fd, bins: LaplaceProbDist(fd, bins)
    else:
        estimator = lambda fd, bins: MLEProbDist(fd)

    tagger = trainer.train_supervised(labeled_sequences, estimator)

    # Get the Frequency Distribution for the language model from assignment 1
    #  and update the current (already trained) HiddenMarkovModelTagger object.
    if args.lm:
        sentences = read_tac()  # Get the list of normalized sentences.
        print "number of retrieved sentences =", len(sentences)

        conditional_samples = get_cond_samples(sentences)  # of the form:[(condition, sample), (condition, sample), ...]
        print "number of conditional samples =", len(conditional_samples)

        conditional_frequency_distribution = ConditionalFreqDist(conditional_samples)
        print "conditional_frequency_distribution =", conditional_frequency_distribution

        transition_probability = ConditionalProbDist(
            cfdist=conditional_frequency_distribution,
            probdist_factory=estimator,
            bins=len(ALPHABET)
        )

        old_transition_probability = tagger.__getattribute__('_transitions')

        # Add the already trained transition probabilities to the new ones:
        for (condition, prob_dist) in transition_probability.iteritems():
            if condition in old_transition_probability:
                new_freq_dist = prob_dist.freqdist() + old_transition_probability[condition].freqdist()
                if type(prob_dist) is LaplaceProbDist:
                    transition_probability[condition] = LaplaceProbDist(new_freq_dist, prob_dist._bins)
                elif type(prob_dist) is MLEProbDist:
                    transition_probability[condition] = MLEProbDist(new_freq_dist)

        # Update the model transitions
        tagger.__getattribute__('_transitions').update(transition_probability)

    ##
    # Prediction
    ##
    print "- - -"
    for i in range(len(x_test)):  # for each sentence in test set
        print "cipher:\t\t", x_test[i]
        print "plain:\t\t", y_test[i]
        # predict the real characters based on the list of ciphered characters:
        predictions = tagger.best_path(list(x_test[i]))  # get array of predicted characters
        print "prediction:\t", ''.join(predictions)
        print "- - -"

    test_sequences = get_sequences(x_test, y_test)  # of the form [ [('#','#')...], [...], ...]
    tagger.test(test_sequences)  # print the accuracy


if __name__ == '__main__':
    main()
