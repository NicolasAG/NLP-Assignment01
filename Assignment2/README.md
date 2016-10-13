#Assignment 02
This is the work I did for the second assignment of this class.
See requirements below:

## Data
See folders `cipher1/` and `cipher2/`
They contain short samples of both ciphered text (i.e., encrypted text) and its associated plaintext (i.e., plain English).

Their are two different ciphers of the sample text:

+ cipher1/
This cipher is a simple letter substitution cipher, where each letter in the plaintext is deterministically replaced by another letter during encryption.
+ cipher2/
This cipher is a more complex cipher, in which there are two letter substitution ciphers. When encrypting each letter, one of the two is randomly chosen, and that cipher is used to generate the ciphered text letter.

The plaintext and ciphered text alphabets are both the 26 letters of the alphabet (lowercase), plus the space, comma, and period, for a total of 29 symbols.

The task is to explore several HMM models for solving these ciphers.
Each HMM sample will be a sentence, and each time step within a sequence will be one character.
The hidden layer will consist of the plaintext characters, while the observed layer will consist of ciphered text characters.

## Standard HMM
Implement a system which trains a standard HMM on the training set using MLE, and tests on the testing data.
You can use NLTK’s `nltk.tag.hmm`.
Your code should run in the following way: `python decipher.py <cipher_folder>`
It should print out the deciphered test set, and report the accuracy score.

## Laplace smoothing
See if smoothing will help improve performance.
Modify your code and add an option that implements Laplace smoothing during training.

It should be possible to turn Laplace smoothing on at the command line in the following way: `python decipher.py -laplace <cipher_folder>`

## Improved plaintext modelling
The training set that is very small. Try to improve performance by having a better model of character bigram transitions in English.
Incorporate this information in the training procedure by pre-processing and getting character transition counts from the samples of English available from Assignment 1 (private data).
These counts should supplement, not replace the counts from the original training set.

Deal with the following issues:
1. Sentence segmentation
2. Lower-casing the text
3. Removing any other character which is not one of the 29 symbols
4. Removing extra space from the beginning and end of each sentence

It should be possible to turn this option on at the command line in the following way: `python decipher.py -lm <cipher_folder>`
In addition, it should be possible to turn on both Laplace smoothing and the improved language modelling.

## Report
Experiment on the two ciphers, reporting accuracy for each of the settings in a table.
Write a brief report on the results, noting whether each change was successful in improving performance.
Were there any surprises or unexpected results? Do you achieve close to perfect accuracy? If not, why not?
Try to explain and speculate on the reasons for these results.
