This is the README file for A0164669M's and A0228393X's submission
Email for A0164669M: e0148711@u.nus.edu
Email for A0228393X: e0673226@u.nus.edu

== Python Version ==

I'm (We're) using Python Version <3.8.8> for
this assignment.

== General Notes about this assignment ==

Give an overview of your program, describe the important algorithms/steps
in your program, and discuss your experiments in general.  A few paragraphs
are usually sufficient.

INDEXING
The major steps involved in the indexing process are
1. preprocess to remove weird code segments, website links, escape characters, attached
    punctuations; then perform case folding, tokenization, stemming
2. log term frequency calculation
3. positional index generation

Pickle object structure for dictionary.txt
{
 'collection_size': size,
 'dictionary': {term: (doc frequency, offset)}
}

Pickle object structure for postings.txt
{
 term:
    {
     docID1: [normalized tf, [1, 2, 3, 4]],
     docID2: [normalized tf, [1, 2, 3, 4]]
    }
}, where [1,2,3,4] are the sorted positional index

SEARCHING
The major steps involved in the searching process are
1. convert the query line into free text query, i.e., ignore AND operator and phrasal query
2. preprocess the query with case folding, stop words removal, punctuation removal, stemming
3. perform query expansion with synonyms from nltk wordnet
    1. 2 additional synonyms for each word from the original query are added to the query
4. query the expanded free text with VSM and use a dictionary doc_score to store the doc scores
5. extract the phrases in the query and perform phrasal query
    1. suppose there are k phrases in the query, and a particular doc satisfies n phrasal
        query, then the score of phrasal query for this document is n / k
    2. use a dictionary 'phrasal_doc_score' to store the doc scores for phrasal query
6. update the doc scores from step 4 with the doc scores from step 5
    1. documents satisfying the phrasal query are given extra scores
    2. normalise the values of doc_score with x = (x - min) / (max - min)
    3. doc_score[doc] += alpha * phrasal_doc_score[doc]
    4. alpha is a hyperparameter to determine how much score should be added;
        currently it is set to 0.1
7. sort the adjusted doc_score and write to the results file

EXPERIMENT
After experiment, we found that the above-mentioned technique on query expansion with WordNet
synonyms does not have positive effects on the performance. We think it might be because such
expansion reduces the query precision and thus gives a larger error. Therefore, we have commented
out the code section that implements it
However, the way we adjust the doc score, and thus the ranking for query results with phrasal
query actually gives a better performance. The exact comparison and scores are shown in the word
document in this submission.

== Files included with this submission ==

List the files in your submission here and provide a short 1 line
description of each file.  Make sure your submission's files are named
and formatted correctly.

== Statement of individual work ==

Please put a "x" (without the double quotes) into the bracket of the appropriate statement.

[x] We, A0164669M and A0228393X, certify that we have followed the CS 3245 Information
Retrieval class guidelines for homework assignments.  In particular, we
expressly vow that we have followed the Facebook rule in discussing
with others in doing the assignment and did not take notes (digital or
printed) from the discussions.

[ ] I/We, A0000000X, did not follow the class rules regarding homework
assignment, because of the following reason:

<Please fill in>

We suggest that we should be graded as follows:

<Please fill in>

== References ==

<Please list any websites and/or people you consulted with for this
assignment and state their role>
stackoverflow sites on the usage of regular expressions