#Assignment 03 - Lesk’s Algorithm

This is the work I did for the third assignment of this class.
See requirements below:
### Word Sense Disambiguation with Lesk’s Algorithm

## Data

Publicly available data set of SemEval 2013 Shared Task#12 (Navigli and Jurgens, 2013)
English dataset can be found in: `./multilingual-all-words.en.xml`
True wordnet lemma_keys can be found in `./wordnet.en.key`
More information on the data set can be found at https://www.cs.york.ac.uk/semeval-2013/task12/index.php%3Fid=data.html

• The gold standard key presents solutions using lemma sense keys, which are distinct from the synset numbers.
Convert between them to perform the evaluation.
This webpage https://wordnet.princeton.edu/man/senseidx.5WN.html explains what lemma sense keys are.

• The data set contains multi-word phrases, which should be resolved as one entity (e.g., latin america).
Make sure that you are converting between underscores and spaces correctly, and check that you are dealing with upper- vs lower-case appropriately.

• For simplicity, we are using instances with id beginning with d001 as the dev set, and the remaining cases as the test set.
This is different from the setting in the original SemEval evaluation, so the results are not directly comparable.

## Models for WSD

NLTK’s interface to WordNet v3.0 as the lexical resource.

Apply word tokenization and lemmatization as necessary, and remove stop words.

1. The most frequent sense baseline: this is the sense indicated as #1 in the synset according to WordNet

2. NLTK’s implementation of Lesk’s algorithm (nltk.wsd.lesk)

3. Combine distributional information about the frequency of word senses, and the standard Lesk’s algorithm.
Justify other parameters:
    + what to include in the sense and context representations
    + how to compute overlap
    + how to trade off the distributional and the Lesk signal: you may use any heuristic, probabilistic model, or other statistical method discussed in class in order to combine these two sources of information.



## Evaluation

Implement the evaluation measures of precision, recall, and F1.

## Report

Discuss the results of your experiments with the three models.
Discuss the successes and difficulties faced by the models.
Include sample output, some analysis, and suggestions for improvements.
The entire report, including the description of your model, must be no longer than two pages.
