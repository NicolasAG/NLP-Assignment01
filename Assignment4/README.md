#Assignment 04 - SumBasic for Multi-document Summarization

This is the work I did for the fourth assignment of this class.
See requirements below:


## Data
Use a news aggregator tool such as Google News to found 4 clusters of articles on the same event or topic.
Each cluster should contain at least 3 articles,
and each article should be of sufficient length to generate an interesting summary from (at least 3-4 paragraphs).

Clean the article text by removing all hyperlinks, formatting, titles and other items that are not the textual body of the articles.
Use any method to do this (including by hand).
Deal with non-ASCII characters: handle them any way you like, including just replacing them by a similar-looking ASCII character.

Save input into text files called docA-B.txt, where A is a positive integer corresponding to the cluster number,
and B is another positive integer corresponding to the article number within that cluster.
ex: `doc1-2.txt` is the second article in the first cluster.
All the documents are inside the folder `./docs`.

## Models for Multi-document Summarization

Implement SumBasic, as it is described in the lecture notes, in order to generate 100-word summaries for each of the document clusters.
Compare these three models:

1. `orig`: The original SumBasic version: including the non-redundancy update of the word scores.
2. `simplified`: A simplified version of SumBasic: holds the word scores constant and does not incorporate the non-redundancy update.
3. `leading`: A simple Heuristic: takes the leading sentences of one of the articles, up until the word length limit is reached.
Decide how to select the article arbitrarily.

Apply standard preprocessing steps on the input documents: sentence segmentation, lemmatization, ignoring stopwords and case distinctions.

Run the code: `python sumbasic.py <method_name> <files>`
It prints the output summary to standard output. If -v flag, also outputs debug flow and summary length.
ex: `python sumbasic.py simplified ./docs/doc1-*.txt > simplified-1.txt`
will run the simplified version of the summarizer on the first cluster, writing the output to a text file called `simplified-1.txt`.

## Report

Discuss quality of each of the three methods.
Does the non-redundancy update work as expected?
How are the methods successful or not successful? 
How would you order the summary sentences with the SumBasic methods, or another extractive summarization approach?
Cover all aspects of summary quality that we discussed in class.
