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


def remove_prefix_num(word):
    """
    Remove the preceding paragraph number in front of the first word of each paragraph e.g. "3Another", "5The"
    very difficult...
    """
    common_order_suffix = ['st', 'nd', 'rd', 'th']
    if word[0].isnumeric() and word[-1].isalpha() and word[-2].isalpha() and word[-2:] not in common_order_suffix:
        for ind, char in enumerate(word):
            if char.isnumeric():
                continue
            else:
                return word[ind:]
    else:
        return word


def remove_attached_punctuation(word):
    """
    Remove punctuations that have not been tokenized and are attached to a term, e.g. '.hello.'
    """
    if word[0] in string.punctuation:
        word = word[1:]

    if word[-1] in string.punctuation:
        word = word[:-1]

    return word


def preprocess(field):

    # remove weird code segment and website link
    cleaned_field = re.sub('//<!\\[CDATA\\[.*?//\\]\\]>', '', field, flags=re.DOTALL)
    cleaned_field = re.sub('<.*?>', '', cleaned_field, flags=re.DOTALL)
    # if cleaned_field != field:
    #     split_cleaned = set(cleaned_field.split())
    #     split_original = set(field.split())
    #     diff = split_cleaned.symmetric_difference(split_original)
    #     diff = ' '.join(diff)
    #     print(diff)


    # remove all escape characters
    escape_chars = ['\xa0', '\n', '\t', '\r', '\'']
    for escape_char in escape_chars:
        cleaned_field = cleaned_field.replace(escape_char, ' ')

    # apply case_folding, sent_tokenize and word_tokenize
    words_by_sent = [word_tokenize(t) for t in sent_tokenize(cleaned_field.lower())]

    # for each word, remove prefix number and suffix punctuation and then apply stemmer
    cleaned_field = []
    for words in words_by_sent:
        for word in words:
            if word not in string.punctuation:
                word = remove_prefix_num(word)
                word = remove_attached_punctuation(word)
                word = stemmer.stem(word)
                if word not in string.punctuation:
                    cleaned_field.append(word)

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

    position_index = {}
    doc_count = 0
    with open(in_csv, newline='') as f:
        reader = csv.reader(f, dialect='excel')
        for row in reader:
            doc_count += 1
            if doc_count % 100 == 0:
                print("%d indexed!"% doc_count)

            if row[0] == 'document_id':
                continue

            # if row[0] == '3751615':
            #     print(row)

            # aggregating title, content, court and date for now; no structure.
            doc_id, date = row[0], [row[3]]
            title, content, court = preprocess(row[1]), preprocess(row[2]), preprocess(row[4])
            doc_terms = title + content + date + court

            # tally term frequency and positions for each term
            doc_term_freq = {}   # {term: freq}
            doc_term_positions = {}  # {term: [1, 2, 3, 4]}
            for pos, term in enumerate(doc_terms):
                if term not in doc_term_freq:
                    doc_term_freq[term] = 1
                    doc_term_positions[term] = [pos]
                else:
                    doc_term_freq[term] += 1
                    doc_term_positions[term].append(pos)

            # compute log tf for each term using doc_term_freq
            for term, freq in doc_term_freq.items():
                doc_term_freq[term] = compute_log_tf(freq)

            # compute document length
            doc_len = normalise_weight(doc_term_freq)

            # append <docID: [normalized tf, [...positions...]]> to position_index
            for term, log_tf in doc_term_freq.items():
                if term not in position_index:
                    position_index[term] = {doc_id: [log_tf/doc_len, doc_term_positions[term]]}
                else:
                    position_index[term][doc_id] = [log_tf/doc_len, doc_term_positions[term]]

    print("sorting!")
    #sort by term and then sort by docIDs for each term in position_index
    position_index = {term: {doc_id: position_index[term][doc_id] for doc_id in sorted(position_index[term].keys())}
                      for term in sorted(position_index.keys())}

    # write to dictionary.txt and postings.txt
    print("writing to file...")
    dictionary = {}
    postings_txt = b''
    offset = 0

    print("Total: %d terms!" % len(position_index))

    count = 0
    with open(out_postings, 'ab') as p:
        for term, posting in position_index.items():
            count += 1
            pickled_posting = pickle.dumps(posting)
            pickled_len = len(pickled_posting)

            dictionary[term] = (len(posting), offset)
            offset += pickled_len
            #postings_txt += pickled_posting
            p.write(pickled_posting)
            if count % 1000 == 0:
                print("%d writen to file!" % count)

    dictionary_txt = {'collection_size': doc_count,
                      'dictionary': dictionary}

    with open(out_dict, 'wb') as d:
        pickle.dump(dictionary_txt, d)
    #
    # with open(out_postings, 'wb') as p:
    #     p.write(postings_txt)


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
