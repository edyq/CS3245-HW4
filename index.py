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
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
csv.field_size_limit(sys.maxsize)


def usage():
    print("usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file")


def preprocess(field):
    cleaned_field = re.sub('//<!\\[CDATA\\[.*?//\\]\\]>', '', field, flags=re.DOTALL)
    escape_chars = ['\xa0', '\n', '\t', '\r']
    for escape_char in escape_chars:
        cleaned_field = cleaned_field.replace(escape_char, ' ')

    words_by_sent = [word_tokenize(t) for t in sent_tokenize(cleaned_field.lower())]
    cleaned_field = [stemmer.stem(word) for words in words_by_sent for word in words if word not in string.punctuation]
    return cleaned_field


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

    with open(in_csv, newline='') as f:
        reader = csv.reader(f, dialect='excel')
        for row in reader:
            if row[0] == 'document_id':
                continue

            doc_id, date = row[0], row[3]
            title, content, court = preprocess(row[1]), preprocess(row[2]), preprocess(row[4])

            # aggregating title, content, court and date for now; no structure.



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
