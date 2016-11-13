"""
Created on Oct 26, 2015

@author: jcheung
@student: Nicolas A.G.
"""

import xml.etree.cElementTree as ET
import codecs

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.wsd import lesk

import treetaggerwrapper as tt  # use same tagger as the MWSD task to get same lemmas (https://www.cs.york.ac.uk/semeval-2013/task12/index.php%3Fid=data.html)

STOP_LIST = set(stopwords.words('english'))
WN_POS_LIST = ['a', 'n', 'v', 'r']
TAGGER = tt.TreeTagger()  # build a TreeTagger wrapper


class WSDInstance:

    def __init__(self, my_id, lemma, pos, context, index):
        """
        Constructor for a WordSenseDisambiguation instance
        Each instance comes from one element
        Each element comes from one sentence
        We have many sentences
        :param my_id: id of the sentence element (unique for ALL elements)
        :param lemma: lemma of the sentence element
        :param context: list of lemmas of all elements in that sentence
        :param index: index of the element in that sentence (unique for that sentence ONLY)
        """
        self.id = my_id         # id of the WSD instance
        self.lemma = lemma      # lemma of the word whose sense is to be resolved
        self.pos = pos          # pos tag of the word whose sense is to be resolved
        self.context = context  # lemma of all the words in the sentential context
        self.index = index      # index of lemma within the context

        self.synset_name = {    # synset name to be resolved based on 3 variations of Lesk's algorithm.
            'lesk1': None,
            'lesk2': None,
            'lesk3': None
        }

    def __str__(self):
        """
        For printing purposes.
        """
        return 'id:%s\nlemma:%s\npos:%s\ncontext:%s\nindex:%d\nsynsets:%s\n' % \
               (self.id, self.lemma, self.pos, self.context, self.index, self.synset_name)


def to_ascii(s):
    """
    remove all non-ascii characters
    """
    return codecs.encode(s, 'ascii', 'ignore')


def synset_to_lemma(synset_name):
    return wn.synset(synset_name).lemmas()[0].key()


def lemma_to_synset(lemma_key):
    return wn.lemma_from_key(lemma_key).synset().name()


def load_instances(f, remove_stop_words=True):
    """
    Parse xml file and load two lists of cases to perform WSD on.
    :param f: xml file to parse that contains all the WSD instances.
    :param remove_stop_words: flag to decide to remove stop words from the context.
    :return: two dictionaries (one for dev, one for test) of the form: {element_id: WSDInstance}
    """
    print "\nLoading instances..."
    tree = ET.parse(f)
    root = tree.getroot()
    
    dev_instances = {}
    test_instances = {}
    
    for text in root:
        if text.attrib['id'].startswith('d001'):
            instances = dev_instances
        else:
            instances = test_instances

        for sentence in text:
            context = [to_ascii(el.attrib['lemma']) for el in sentence]  # sentence context = [element_lemma1, element_lemma2, ...]
            if remove_stop_words:
                # filter out stop words
                context = filter(lambda w: w not in STOP_LIST and w != '@card@', context)
            for i, el in enumerate(sentence):  # for each element, construct a WSDInstance
                if el.tag == 'instance':
                    my_id = el.attrib['id']  # element_id
                    lemma = to_ascii(el.attrib['lemma'])  # element_lemma
                    pos = el.attrib['pos'][0].lower()  # element_pos converted to wordnet POS: ['n', 'v', 'a', 'r']
                    instances[my_id] = WSDInstance(my_id, lemma, pos, context, i)
    print "done."
    print "dev_instances:", len(dev_instances)
    print "test_instances:", len(test_instances)
    return dev_instances, test_instances


def load_key(f, lemma2synset=True):
    """
    Read file f and load the true sense(s) for each element id.
    :param f: file containing all senses for each element.
    :param lemma2synset: flag to decide to directly convert from lemma_key to synset_name
    :return: two dictionaries (one for dev, one for test) of the form: {element_id: [sense1_key, sense2_key, ...]}
    """
    print "\nLoading true senses..."
    dev_key = {}
    test_key = {}
    for line in open(f):
        if len(line) <= 1:
            continue
        doc, my_id, sense_key = line.strip().split(' ', 2)
        if lemma2synset:
            # convert lemma keys to their corresponding synset name
            sense_key = [lemma_to_synset(lemma_key) for lemma_key in sense_key.split()]
        if doc == 'd001':
            dev_key[my_id] = sense_key
        else:
            test_key[my_id] = sense_key
    print "done."
    print "dev_key:", len(dev_key)
    print "test_key:", len(test_key)
    return dev_key, test_key


def predict_synsets(wsd_instance, use_pos=True):
    lesk1_synset = lesk_1(wsd_instance, use_pos)
    if lesk1_synset:
        wsd_instance.synset_name['lesk1'] = lesk1_synset.name()

    lesk2_synset = lesk_2(wsd_instance, use_pos)
    if lesk2_synset:
        wsd_instance.synset_name['lesk2'] = lesk2_synset.name()

    lesk3_synset = lesk_3(wsd_instance, use_pos)
    if lesk3_synset:
        wsd_instance.synset_name['lesk3'] = lesk3_synset.name()


def lesk_1(wsd_instance, use_pos):
    """
    Most frequent sense baseline: this is the sense indicated as #1 in the synset according to WordNet
    :param wsd_instance: the WSD instance to define.
    :param use_pos: flag to decide to use POS tags to retrieve the synset.
    :return: the first wordnet synset corresponding to that wsd_instance lemma.
    """
    if use_pos and wsd_instance.pos in WN_POS_LIST:
        synsets = wn.synsets(wsd_instance.lemma, wsd_instance.pos)
    else:
        synsets = wn.synsets(wsd_instance.lemma)

    sense2count = {}  # dictionary of the form {synset_name: count, ...}
    for ss in synsets:
        freq = 0
        for lemma in ss.lemmas():
            freq += lemma.count()
        sense2count[ss.name()] = freq

    max_count = 0
    ss = synsets[0]
    for ss_name, count in sense2count.iteritems():
        if count > max_count:
            max_count = count
            ss = wn.synset(ss_name)

    return ss


def lesk_2(wsd_instance, use_pos):
    """
    NLTK's implementation of Lesk's algorithm
    :param wsd_instance: the WSD instance to define.
    :param use_pos: flag to decide to use POS tags to retrieve the synset.
    :return: the wordnet synset returned by nltk.wsd.lesk corresponding to that wsd_instance lemma.
    """
    if use_pos and wsd_instance.pos in WN_POS_LIST:
        return lesk(wsd_instance.context, wsd_instance.lemma, wsd_instance.pos)
    else:
        return lesk(wsd_instance.context, wsd_instance.lemma)


def lesk_3(wsd_instance, use_pos):
    """
    Combination of distributional information about the word sense frequency and the standard Lesk's algorithm.
    :param wsd_instance: the WSD instance to define.
    :param use_pos: flag to decide to use POS tags to retrieve the synset.
    :return: the most probable wordnet synset according to a combination of distributional and Lesk signals.
    """
    if use_pos and wsd_instance.pos in WN_POS_LIST:
        synsets = wn.synsets(wsd_instance.lemma, wsd_instance.pos)
    else:
        synsets = wn.synsets(wsd_instance.lemma)

    context = filter(lambda w: w not in STOP_LIST and w != '@card', wsd_instance.context)  # remove stop words
    context = set(context)  # unique words(lemmas) in context

    sense2count_intersection = {}  # dictionary of the form {synset_name: (count, intersection_size), ...}
    for ss in synsets:
        # Get sense counts
        freq = 0
        for lemma in ss.lemmas():
            freq += lemma.count()

        # Get the context intersection count
        definition = ss.definition()
        tags = tt.make_tags(TAGGER.tag_text(definition))  # tags of the definition
        lemmas = [tag.lemma for tag in tags]  # lemmatize the definition
        intersection = len(context.intersection(lemmas))  # number of intersecting lemmas between definition and context

        sense2count_intersection[ss.name()] = (freq, intersection)

    max_count = 0
    ss = synsets[0]
    alpha = 1.
    beta = 0.5
    for ss_name, (count, intersection) in sense2count_intersection.iteritems():
        combination = alpha*count + beta*intersection
        if combination > max_count:
            max_count = combination
            ss = wn.synset(ss_name)

    return ss


def precision(instances, keys):
    """
    Compute the percentage of predicted synsets that are actually true.
    :param instances: dictionary of the form {id:wsd_instance, } where each wsd_instance has a predicted synset.
    :param keys: dictionary of the form {id:[synset1_name, ...], }
    :return: a dictionary of the form {'lesk1':precision1, 'lesk2':precision2, 'lesk3':precision3}
    """
    assert len(instances) == len(keys)
    precision1_count = 0.
    precision2_count = 0.
    precision3_count = 0.

    for id, wsd_instance in instances.iteritems():
        # print keys[id]
        # print wsd_instance
        if wsd_instance.synset_name['lesk1'] and wsd_instance.synset_name['lesk1'] in keys[id]:
            precision1_count += 1
        if wsd_instance.synset_name['lesk2'] and wsd_instance.synset_name['lesk2'] in keys[id]:
            precision2_count += 1
        if wsd_instance.synset_name['lesk3'] and wsd_instance.synset_name['lesk3'] in keys[id]:
            precision3_count += 1

    return {
        'lesk1': (precision1_count / len(instances))*100,
        'lesk2': (precision2_count / len(instances))*100,
        'lesk3': (precision3_count / len(instances))*100
    }


def recall(instances, keys):
    """
    Compute the percentage of true synsets that were predicted.
    :param instances: dictionary of the form {id:wsd_instance, } where each wsd_instance has a predicted synset.
    :param keys: dictionary of the form {id:[synset1_name, ...], }
    :return: a dictionary of the form {'lesk1':precision1, 'lesk2':precision2, 'lesk3':precision3}
    """
    assert len(instances) == len(keys)
    recall1_count = 0.
    recall2_count = 0.
    recall3_count = 0.

    for id, synsets in keys.iteritems():
        if instances[id].synset_name['lesk1'] and instances[id].synset_name['lesk1'] in synsets:
            recall1_count += 1
        if instances[id].synset_name['lesk2'] and instances[id].synset_name['lesk2'] in synsets:
            recall2_count += 1
        if instances[id].synset_name['lesk3'] and instances[id].synset_name['lesk3'] in synsets:
            recall3_count += 1

    return {
        'lesk1': (recall1_count / len(keys)) * 100,
        'lesk2': (recall2_count / len(keys)) * 100,
        'lesk3': (recall3_count / len(keys)) * 100
    }


def f1(p, r):
    """
    Compute the F1-score from precision and recall scores
    :param p: precision for each Lesk's algorithms
    :param r: recall for each Lesk's algorithms
    :return: f1-score for each Lesk's algorithms
    """
    f1_lesk1 = 0.
    f1_lesk2 = 0.
    f1_lesk3 = 0.

    if p['lesk1'] + r['lesk1'] > 0:
        f1_lesk1 = (2 * p['lesk1'] * r['lesk1']) / (p['lesk1'] + r['lesk1'])

    if p['lesk2'] + r['lesk2'] > 0:
        f1_lesk2 = (2 * p['lesk2'] * r['lesk2']) / (p['lesk2'] + r['lesk2'])

    if p['lesk3'] + r['lesk3'] > 0:
        f1_lesk3 = (2 * p['lesk3'] * r['lesk3']) / (p['lesk3'] + r['lesk3'])

    return {
        'lesk1': f1_lesk1,
        'lesk2': f1_lesk2,
        'lesk3': f1_lesk3,
    }


if __name__ == '__main__':
    data_f = 'multilingual-all-words.en.xml'
    key_f = 'wordnet.en.key'
    dev_instances, test_instances = load_instances(data_f, remove_stop_words=True)
    dev_keys, test_keys = load_key(key_f, lemma2synset=True)  # directly do the conversion from lemma_key to synset_name

    # IMPORTANT: keys contain fewer entries than the instances; need to remove them
    print "\nFiltering instances to make sure we have the true senses..."
    dev_instances = {k: v for (k, v) in dev_instances.iteritems() if k in dev_keys}
    test_instances = {k: v for (k, v) in test_instances.iteritems() if k in test_keys}
    print "done."
    print "new dev_instances:", len(dev_instances)
    print "new test_instances:", len(test_instances)

    ###
    # PREDICT THE SYNSET FOR EACH WSD_INSTANCES
    ###
    print "\nRunning 3 variations of Lesk's algorithm to predict the word sense of each wsd_instances..."
    for wsd_instance in dev_instances.values():
        predict_synsets(wsd_instance, use_pos=True)

    for wsd_instance in test_instances.values():
        predict_synsets(wsd_instance, use_pos=True)
    print "done."

    ###
    # EVALUATE EACH MODEL
    ###
    print "\nEvaluation of dev_set:"
    p_dev = precision(dev_instances, dev_keys)
    r_dev = recall(dev_instances, dev_keys)
    f1_dev = f1(p_dev, r_dev)
    print "precision:", p_dev
    print "recall:", r_dev
    print "f1:", f1_dev

    print "\nEvaluation of test_set:"
    p_test = precision(test_instances, test_keys)
    r_test = recall(test_instances, test_keys)
    f1_test = f1(p_test, r_test)
    print "precision:", p_test
    print "recall:", r_test
    print "f1:", f1_test
