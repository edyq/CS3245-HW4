#!/usr/bin/python3
import re
import nltk
import os
import sys
import math
import getopt
import pickle
import string
from nltk.tokenize import sent_tokenize, TweetTokenizer
tknzr = TweetTokenizer()
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


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


def build_index(in_dir, out_dict, out_postings):
    """
    build index from documents stored in the input directory,
    then output the dictionary file and postings file
    """
    print('indexing...')

    # compute term frequency and term weights for each document and add into the dictionary of postings
    postings_dict = {}
    all_document_ids = sorted([int(doc_id) for doc_id in os.listdir(in_dir)])
    for doc_id in all_document_ids:
        doc_dir = os.path.join(in_dir, str(doc_id))
        with open(doc_dir, 'r') as f:
            term_freq_dict = {}

            for line in f:
                tokens = preprocess(line)
                for token in tokens:
                    if token not in term_freq_dict:
                        term_freq_dict[token] = 1
                    else:
                        term_freq_dict[token] += 1

            # compute and store 1 + log(tf)
            for term, freq in term_freq_dict.items():
                term_freq_dict[term] = compute_log_tf(freq)

            # compute document length
            doc_len = normalise_weight(term_freq_dict)

            # store in postings in the format of (doc_id, normalized_weight)
            for term, log_tf in term_freq_dict.items():
                if term not in postings_dict:
                    postings_dict[term] = [(doc_id, log_tf/doc_len)]
                else:
                    postings_dict[term].append((doc_id, log_tf/doc_len))

    # sort the postings dict by term
    sorted_postings_dict = {key: postings_dict[key] for key in sorted(postings_dict.keys())}

    # output to dictionary.txt and postings.txt
    dictionary_txt = {'collection_size': len(all_document_ids)}
    offset_dict = {}
    postings_txt = b''
    offset = 0

    for term, posting in sorted_postings_dict.items():
        pickled_posting = pickle.dumps(posting)
        pickled_len = len(pickled_posting)

        # store offset, length of pickle object, and document frequency in the dictionary
        offset_dict[term] = (offset, pickled_len, len(posting))
        offset += pickled_len
        postings_txt += pickled_posting

    # output to files (store all document IDs for processing NOT queries in dictionary.txt as well)
    dictionary_txt['dictionary'] = offset_dict

    with open(out_dict, 'wb') as d:
        pickle.dump(dictionary_txt, d)

    with open(out_postings, 'wb') as p:
        p.write(postings_txt)


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
