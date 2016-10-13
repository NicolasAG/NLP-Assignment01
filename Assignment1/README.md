#Assignment 01

This is the work I did for the first assignment of this class.
See requirements below:

## Data

Available from the course website (http://cs.mcgill.ca/~jcheung/teaching/fall-2016/comp599/assignments/a1/a1data.zip).
Protected with a password because private.

This corpus is a collection of news articles that were used in the TAC 2010 and 2011 summarization data sets, and contains articles separated into five categories:

1. Accidents and Natural Disasters

2. Attacks

3. Health and Safety

4. Endangered Resources

5. Investigations and Trials

The task is to train a document classification system to distinguish documents from these five topic categories.

The raw text files are stored in the subfolders /tac2010 and /tac2011. Included in each data set is also a file that stores the topic category (i.e., the label to be predicted), for example /tac2010/tac2010.labels, where each line is in the format of “document id tab category”. The category will be an integer from 1 to 5, corresponding to the five categories listed above.


## Preprocessing and feature extraction

Preprocess the input documents to extract feature vector representations of them. The features should be N-gram counts, for N ≤ 2.
Use NLTK’s tokenizer to help you in this task. You may also use scikitlearn’s feature extraction module.
You should experiment with the complexity of the N-gram features (i.e., unigrams, or unigrams and bigrams), whether to distinguish upper and lower case, whether to remove stop words, etc.
NLTK contains a list of stop words in English. You may choose to experiment with the amount of smoothing/regularization in training the models to achieve better results, though you can also just leave these at the default settings.

## Model selection
Use TAC 2010 as your training set and TAC 2011 as your testing set.
You should set aside a small portion of your training set (around 20%) as a development set for tuning your models.
Compare the logistic regression, support vector machine (with a linear kernel), and Naive Bayes algorithms.

## Report
Write a short report on your method and results, carefully document the range of parameter settings that you tried and your experimental procedure.
It should be no more than one page long.
Report on the performance in terms of accuracy, and speculate on the successes and failures of the models.
Which machine learning classifer produced the best performance?
For the overall best performing model, include a confusion matrix as a form of error analysis.
Also, explain the role of the development set in the above experiment.
