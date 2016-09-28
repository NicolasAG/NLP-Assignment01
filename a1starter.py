"""
Created on Jul 14, 2015

@author: jcheung
"""

import os
import codecs
import numpy as np
import matplotlib.pyplot as plt

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm, linear_model, naive_bayes
from sklearn.metrics import confusion_matrix


wordnet_lemmatizer = WordNetLemmatizer()

# model parameters
LOWERCASE = False
STOP_LIST = set(stopwords.words('english'))
REMOVE_STOP = False
REMOVE_PUNCT = False
LEMMATIZE = False
N_GRAM = 1


################################################################################
# reading data set
def read_tac(year, vectorizer):
    """
    Read data set and return feature matrix X and labels y.
    
    X - (ndocs x n_features)
    Y - (ndocs)
    """
    # modify this according to your directory structure
    sub_folder = '../data/tac%s' % year
    X, Y = [], []
    
    # labels
    labels_f = 'tac%s.labels' % year
    
    fh = open(os.path.join(sub_folder, labels_f))
    for line in fh:
        doc_id, label = line.split()
        Y.append(int(label))
    
    # tac 10
    if year == '2010':
        template = 'tac10-%04d.txt'
        s, e = 1, 921
    elif year == '2011':
        template = 'tac11-%04d.txt'
        s, e = 921, 1801
    else:
        print "no data for the year ", year
        return

    for i in xrange(s, e):
        file_name = os.path.join(sub_folder, template % i)
        X.append(extract_tokens(file_name))
    # X is now a 2D matrix of tokens.

    corpus = [" ".join(tokens) for tokens in X]  # join each token by a space for each example
    if year == "2010":  # fit the features on training data (2010)
        X = vectorizer.fit_transform(corpus)
    elif year == "2011":  # only extract features on validation and test data (2011)
        X = vectorizer.transform(corpus)
    else:
        print "no data for the year ", year
        return

    """
    n_features = 100  # TODO: you'll have to figure out how many features you need: need max(X[i][j]) for all i,j
    # convert indices to numpy array
    for j, x in enumerate(X):
        arr = np.zeros(n_features)
        for feature_value in x:
            arr[feature_value] += 1.0
        X[j] = arr
    """

    Y = np.array(Y)  # make a 1D numpy array
    X = X.toarray()
    # X = np.vstack(tuple(X))  # make a 2D numpy array
    return X, Y
        

################################################################################
# feature extraction
def ispunct(some_string):
    return not any(char.isalnum() for char in some_string)


def get_tokens(s):
    """
    Tokenize into words in sentences.
    
    Returns list of strings
    """
    words = []
    sentences = sent_tokenize(s)
    
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        words.extend(tokens)
    return words


def extract_tokens(f, lowercase=LOWERCASE, remove_stop=REMOVE_STOP, remove_punct=REMOVE_PUNCT,
                   lemmatize=LEMMATIZE):
    """
    Extract features from text file f into a token vector.
    :lowercase: (boolean) whether or not to lowercase everything
    :remove_stop: (boolean) whether or not to remove stop words
    :remove_punct: (boolean) whether or not to remove punctuations
    :lemmatize: (boolean) whether or not to lemmatize
    :n: maximum length of N-grams
    :return: an array of tokens
    """
    s = codecs.open(f, 'r', encoding='utf-8').read()
    s = codecs.encode(s, 'ascii', 'ignore')
    
    tokens = get_tokens(s)
    print "file ", f, " has ", len(tokens), " tokens."
    print tokens

    if lowercase:
        for i, w in enumerate(tokens):
            tokens[i] = w.lower()

    if remove_stop:
        tokens = [w for w in tokens if w not in STOP_LIST]

    if remove_punct:
        tokens = [w for w in tokens if not ispunct(w)]

    if lemmatize:
        def get_wordnet_pos(treebank_pos):
            """
            Util function to convert from Treebank POS tag to WordNet POS tag.
            :param treebank_pos: the Treebank POS tag
            :return: the corresponding WordNet POS tag
            """
            if treebank_pos.startswith('J'):
                return wordnet.ADJ
            elif treebank_pos.startswith('V'):
                return wordnet.VERB
            elif treebank_pos.startswith('N'):
                return wordnet.NOUN
            elif treebank_pos.startswith('R'):
                return wordnet.ADV
            else:
                return None

        pos_tokens = nltk.pos_tag(tokens)  # get Treebank POS tags for the tokens
        for i, (w, pos) in enumerate(pos_tokens):
            pos = get_wordnet_pos(pos)  # convert the POS tag
            if pos:
                tokens[i] = wordnet_lemmatizer.lemmatize(w, pos)
            else:  # if no conversion found, use default.
                tokens[i] = wordnet_lemmatizer.lemmatize(w)

    print tokens
    print "file ", f, " now has ", len(tokens), " tokens."

    return tokens


################################################################################
# evaluation code
def accuracy(gold, predict):
    assert len(gold) == len(predict)
    corr = 0
    for i in xrange(len(gold)):
        if int(gold[i]) == int(predict[i]):
            corr += 1
    acc = float(corr) / len(gold)
    # print 'Accuracy %d / %d = %.4f' % (corr, len(gold), acc)
    return acc
################################################################################


################################################################################
# plot confusion matrix
def plot_confusion_matrix(cm, title='Confusion matrix'):
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(5)
    plt.xticks(tick_marks, (1,2,3,4,5), rotation=45)
    plt.yticks(tick_marks, (1,2,3,4,5))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
################################################################################


if __name__ == '__main__':

    vectorizer = CountVectorizer(ngram_range=(1, N_GRAM))

    X_train, Y_train = read_tac('2010', vectorizer)
    print X_train.shape  # (920,100)
    print Y_train.shape  # (920,)

    X_test, Y_test = read_tac('2011', vectorizer)
    X_validation, Y_validation = X_test[:200], Y_test[:200]  # validation set = first 200 test set
    print X_validation.shape  # (200,100)
    print Y_validation.shape  # (200,)

    X_test, Y_test = X_test[200:], Y_test[200:]  # test set is the following 680 elements
    print X_test.shape  # (680,100)
    print Y_test.shape  # (680,)

    """
    VALIDATION PASS:
    """
    C_values = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1., 10, 1e2, 1e3, 1e4, 1e5]
    A_values = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1., 10, 1e2, 1e3, 1e4, 1e5]
    #######################
    # LOGISTIC REGRESSION #
    #######################
    lr_scores = []
    for c in C_values:
        lr = linear_model.LogisticRegression(C=c)
        lr.fit(X_train, Y_train)  # train based on X_train, Y_train
        # predict based on X_validation
        lr_prediction = lr.predict(X_validation)
        lr_accuracy = accuracy(Y_validation, lr_prediction)
        lr_scores.append(lr_accuracy)
        print "Logistic regression(C=%f) accuracy: %f" % (c, lr_accuracy)
    best_lr_score = max(lr_scores)
    best_lr_c = C_values[lr_scores.index(best_lr_score)]

    #######
    # SVM #
    #######
    svm_scores = []
    for c in C_values:
        clf = svm.SVC(C=c)
        clf.fit(X_train, Y_train)  # train based on X_train, Y_train
        # predict based on X_validation
        svm_prediction = clf.predict(X_validation)
        svm_accuracy = accuracy(Y_validation, svm_prediction)
        svm_scores.append(svm_accuracy)
        print "SVM(C=%f) accuracy: %f" % (c, svm_accuracy)
    best_svm_score = max(svm_scores)
    best_svm_c = C_values[svm_scores.index(best_svm_score)]

    ###############
    # Naive Bayes #
    ###############
    # Multinomial
    multinomial_nb_scores = []
    for a in A_values:
        nb = naive_bayes.MultinomialNB(alpha=a)
        nb.fit(X_train, Y_train)  # train based on X_train, Y_train
        # predict based on X_validation
        nb_prediction = nb.predict(X_validation)
        nb_accuracy = accuracy(Y_validation, nb_prediction)
        multinomial_nb_scores.append(nb_accuracy)
        print "Multinomial Naive Bayes(alpha=%f) accuracy: %f" % (a, nb_accuracy)
    best_multinomial_nb_score = max(multinomial_nb_scores)
    multinomial_nb_alpha = A_values[multinomial_nb_scores.index(best_multinomial_nb_score)]

    """
    TEST PASS
    """
    # Pick the best algorithm and evaluate on test set.
    prediction = []
    if max(best_lr_score, best_svm_score, best_multinomial_nb_score) == best_svm_score:
        # predict using SVM(c) based on X_test
        print "Using SVM(c=%f) with validation score: %f" % (best_svm_c, best_svm_score)
        clf = svm.SVC(C=best_svm_c)
        clf.fit(X_train, Y_train)
        prediction = clf.predict(X_test)
    elif max(best_lr_score, best_svm_score, best_multinomial_nb_score) == best_lr_score:
        # predict using LogisticRegression(c) based on X_test
        print "Using LogisticRegression(c=%f) with validation score: %f" % (best_lr_c, best_lr_score)
        lr = linear_model.LogisticRegression(C=best_lr_c)
        lr.fit(X_train, Y_train)
        prediction = lr.predict(X_test)
    elif max(best_lr_score, best_svm_score, best_multinomial_nb_score) == best_multinomial_nb_score:
        # predict using MultinomialNaiveBayes(a) based on X_test
        print "Using Multinomial Naive Bayes(alpha=%f) with validation score: %f" % (multinomial_nb_alpha, best_multinomial_nb_score)
        nb = naive_bayes.MultinomialNB(alpha=multinomial_nb_alpha)
        nb.fit(X_train, Y_train)
        prediction = nb.predict(X_test)

    score = accuracy(Y_test, prediction)
    print "final score: ", score

    cm = confusion_matrix(Y_test, prediction)
    np.set_printoptions(precision=2)
    print "Confusion matrix, without normalization"
    print cm
    plt.figure()
    plot_confusion_matrix(cm)
    # Normalize the confusion matrix by row (i.e by the number of samples in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    plt.show()
