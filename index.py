#!/usr/bin/python3
import re
import nltk
import os
import sys
import math
import getopt
import pickle
import string
import csv
from nltk.tokenize import sent_tokenize, TweetTokenizer
tknzr = TweetTokenizer()
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
csv.field_size_limit(sys.maxsize)

def usage():
    print("usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file")


def is_word(word):
    """
    Check if a word is a valid term. The word is valid only if it does not contain more than one non-alphabet.
    """
    num_non_alphabets = 0
    for char in word:
        if not char.isalpha():
            num_non_alphabets += 1

    if num_non_alphabets > 1:
        return False
    else:
        if len(word) == 1 and num_non_alphabets == 1:
            return False
        else:
            return True


def preprocess(line):
    """
    Perform case-folding, tokenization and stemming for a line.
    """
    words_by_sent = [tknzr.tokenize(t) for t in sent_tokenize(line.lower())]
    stemmed_words = [stemmer.stem(word) for words in words_by_sent for word in words]

    # remove punctuations and numbers
    return [word for word in stemmed_words if is_word(word)]


def compute_log_tf(freq):
    """
    Compute the log_tf weight.
    """
    if freq > 0:
        return 1 + math.log(freq, 10)
    else:
        return 0


def normalise_weight(term_weight_dict):
    """
    Compute the document length.
    """
    sum = 0
    for term, weight in term_weight_dict.items():
        sum += math.pow(weight, 2)

    return math.sqrt(sum)


def build_index(in_csv, out_dict, out_postings):
    """
    build index from documents stored in the input directory,
    then output the dictionary file and postings file
    """
    print('indexing...')

    count = 0
    smaller_csv = []
    with open(in_csv, newline='') as f:
        reader = csv.reader(f, dialect='excel')
        for row in reader:
            count += 1
            if count == 1:
                smaller_csv.append(row)
            elif count > 17145:
                smaller_csv.append(row)
            elif count % 200 == 0:
                smaller_csv.append(row)

    with open('smaller_dataset.csv', 'w', newline='') as w:
        writer = csv.writer(w)
        writer.writerows(smaller_csv)


input_directory = output_file_dictionary = output_file_postings = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-i': # input directory
        input_directory = a
    elif o == '-d': # dictionary file
        output_file_dictionary = a
    elif o == '-p': # postings file
        output_file_postings = a
    else:
        assert False, "unhandled option"

if input_directory == None or output_file_postings == None or output_file_dictionary == None:
    usage()
    sys.exit(2)

build_index(input_directory, output_file_dictionary, output_file_postings)
