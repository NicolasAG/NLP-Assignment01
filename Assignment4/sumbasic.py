import codecs
import argparse
import random
import math

from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer


STOP_LIST = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


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


def ispunct(some_string):
    return not any(char.isalnum() for char in some_string)


def preprocess_sentence(sentence):
    """
    Process some sentence string: put to lower case, remove stop words & punctuation and lemmatize.
    :param sentence: the string to process.
    :return: the new string
    """
    p_line = sentence.lower()  # put to lower case
    p_line = word_tokenize(p_line.strip())  # split by words
    p_line = pos_tag(p_line)  # get POS tags
    for i, (w, pos) in enumerate(p_line):
        pos = get_wordnet_pos(pos)  # convert the POS tag
        if pos:
            p_line[i] = LEMMATIZER.lemmatize(w, pos)
        else:  # if no conversion found, use default.
            p_line[i] = LEMMATIZER.lemmatize(w)
    p_line = [w for w in p_line if w not in STOP_LIST and not ispunct(w)]  # remove stop words and punctuation
    return ' '.join(p_line)


def preprocess_file(sentences):
    """
    Process some sentences but keep track of the originals.
    :param sentences: array of sentences to process.
    :return: dictionary of preprocessed sentences of the form: {i:[original, preprocessed], ...}
    """
    d = {}  # dictionary to return
    for i, o_line in enumerate(sentences):
        p_line = preprocess_sentence(o_line)
        d[i] = [o_line, p_line]
    return d


def merge_files_content(files_content):
    """
    Make a dictionary from sentence_id to [original_version, processed_version].
    :param files_content: list of sentence dictionaries for each document.
    :return: a dictionary of the form: {id:[original_sentence, processed_sentence], ...}
    """
    d = {}
    for file_number, content in enumerate(files_content):  # for each file,
        # content ~ {<#>: [original, processed], ...}
        for sent_number, sentence in content.iteritems():  # for each sentence,
            # sentence ~ [original, processed]
            d[str(file_number) + '_' + str(sent_number)] = sentence
    return d


def get_word_probas(sentences):
    """
    Compute word probabilities.
    :param sentences: dictionary from sentence_id to [original_version, processed_version].
    :return: dictionary of word probabilities from word to probability.
    """
    word_probas = {}
    total = 0.
    for sentence in sentences.values():  # for each sentence,
        # sentence ~ [original, processed]
        words = sentence[1].split()  # consider the preprocessed sentence
        for w in words:  # for each word lemma, update count.
            total += 1.
            if w in word_probas:
                word_probas[w] += 1.
            else:
                word_probas[w] = 1.
    for w in word_probas.keys():
        word_probas[w] /= total
    return word_probas


def get_sentence_scores(sentences, word_probas):
    """
    Compute each sentence average word probability.
    :param sentences: dictionary from sentence_id to [original_version, processed_version].
    :param word_probas: dictionary from word to its probability.
    :return: dictionary from score to sentence_id
    """
    sentence_scores = {}  # dictionary from score to sentence_id.
    for sent_id, sentence in sentences.iteritems():  # for each sentence,
        # sentence ~ [original, processed]
        words = sentence[1].split()  # consider preprocessed sentence.
        # score = sum of the probabilities of each word DIVIDED BY total number of words in sentence.
        score = math.fsum([word_probas[w] for w in words]) / len(words)
        if score in sentence_scores:
            sentence_scores[score].append(sent_id)
            # print "multiple sentences for the same score! %f" % score
            # print sentence_scores[score]
        else:
            sentence_scores[score] = [sent_id]
    return sentence_scores


def sumbasic(files_content, non_redundancy=True, limit=100, v=False):
    """
    Build a summary of the different files using the SumBasic algorithm.
    :param files_content: list of dictionaries for each document.
    :param non_redundancy: flag to decide to include the word-score update.
    :param limit: the number of words to aim for in the summary.
    :param v: flag for verbose output.
    :return: a 100-word summary for the corpus of documents.
    """
    if non_redundancy and v:
        print "\nGenerating `orig` summary..."
    elif v:
        print "\nGenerating `simplified` summary..."

    # Get ONE dictionary of the form {sentence_id: [original_version, processed_version], ...} for ALL files
    sentences = merge_files_content(files_content)

    # Get the word probabilities
    word_probas = get_word_probas(sentences)

    # Get the sentence scores: of the form {score:[sentence1_id,...], ...}
    sentence_scores = get_sentence_scores(sentences, word_probas)

    summary = ""
    while len(summary) < limit:
        # print "max score: %f" % max(sentence_scores.keys())
        # print "sentences:", sentence_scores[max(sentence_scores.keys())]
        sent_ids = sentence_scores[max(sentence_scores.keys())]
        sent_id = sent_ids[random.randint(0, len(sent_ids)-1)]  # if multiple ids for the same score: take a random one.
        summary += sentences[sent_id][0] + ' '  # [0] : add the original sentence, not the processed one [1].
        if non_redundancy:
            # update word probabilities
            for w in sentences[sent_id][1].split():  # consider preprocessed sentence for non-redundancy update.
                word_probas[w] *= word_probas[w]  # square the probability
            # update sentence scores
            sentence_scores = get_sentence_scores(sentences, word_probas)

    print summary
    if v: print "length: %d" % len(summary)
    return summary


def leading(files_content, limit=100, v=False):
    """
    Build a summary of different files by taking the first few sentences of one file.
    :param files_content: list of dictionaries for each document.
    :param limit: the number of words to aim for in the summary.
    :param v: flag for verbose output.
    :return: a 100-word summary for the corpus of documents.
    """
    if v: print "\nGenerating `leading` summary..."
    summary = ""
    file_number = random.randint(0, len(files_content)-1)  # take a random file
    if v: print "using file %d" % file_number
    for i, sentence in files_content[file_number].iteritems():
        # sentence = [original_line, processed_line]
        if len(summary) < limit:
            summary += sentence[0]+' '  # take the original sentence
        else:
            break
    print summary
    if v: print "length: %d" % len(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Multi-document Summarization.")
    parser.add_argument("method",
                        choices=("orig", "simplified", "leading"),
                        help="which model to run: one of `orig`, `simplified`, or `leading`.")
    parser.add_argument("cluster",
                        nargs='+',
                        help="path to a cluster. ex: `./docs/doc1-*.txt`")
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="increase output verbosity")
    args = parser.parse_args()
    if args.verbose: print args

    if args.verbose: print "\nProcessing files..."
    files_content = []  # list of dictionaries for each document.
    for f_name in args.cluster:
        if args.verbose: print "%s..." % f_name
        sentences = codecs.open(f_name, 'r', encoding='utf-8').read()
        sentences = codecs.encode(sentences, 'ascii', 'ignore')
        sentences = sent_tokenize(sentences)
        if args.verbose: print "... has %d sentences." % len(sentences)
        files_content.append(preprocess_file(sentences))
    if args.verbose: print "done."
    # files_content is of the form: [
    #   {
    #     0:['original_FIRST_sentence', 'processed_FIRST_sentence'],
    #     1:['original_SECOND_sentence', 'processed_SECOND_sentence'],
    #     ...
    #   },
    #   ...
    # ]

    if args.method == 'orig':
        sumbasic(files_content, non_redundancy=True, limit=100, v=args.verbose)
    elif args.method == 'simplified':
        sumbasic(files_content, non_redundancy=False, limit=100, v=args.verbose)
    elif args.method == 'leading':
        leading(files_content, limit=100, v=args.verbose)


if __name__ == '__main__':
    main()
