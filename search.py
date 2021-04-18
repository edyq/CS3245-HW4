#!/usr/bin/python3
import sys
import math
import getopt
import pickle
import heapq
from nltk import PorterStemmer
from nltk import word_tokenize
from collections import defaultdict
from collections import Counter

dictionary = {}


def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q query-file -o output-file-of-results")


def run_search(dict_file, postings_file, query_file, results_file):
    print('running search on the query...')
    # TODO: make use of relevant docs provided in the query

    global dictionary
    dic_file = open(dict_file, 'rb')
    dictionary = pickle.load(dic_file)
    dic_file.close()

    query_line = open(query_file, 'r').readline().rstrip()
    query_tokens = process_query(query_line)
    doc_score = {}
    for query in query_tokens:
        if isinstance(query[0], list):
            new_score = query_phrase(query)
        else:
            new_score = query_free_text(query)
        doc_score = update_doc_score(new_score)

    # sth like this
    result = ' '.join(map(str, [k for k, v in sorted(doc_score.items(), key=lambda item: item[1], reverse=True)]))
    out_file = open(results_file, 'w')
    out_file.write(result)
    out_file.close()


def process_query(query):
    """
    :param query: string
    :return: list
    phrasal queries are enclosed in List object again
    return processed_query = [[[]], [], []]
    where processed_query[0] stands for phrasal query
    processed_query[1] and processed_query[2] are free text queries
    AND relation between the elements of the outer list
    """
    query_list = query.split(' AND ')
    processed_query = []
    for query in query_list:
        if query[0] == '"' and query[-1] == '"':
            query = query[1: -1]
            tokens = tokenize(query)
            processed_query.append([tokens])
        else:
            tokens = tokenize(query)
            processed_query.append(tokens)
    return processed_query


def tokenize(query):
    tokens = word_tokenize(query)
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token.lower()) for token in tokens]
    return [token for token in stemmed_tokens if token.isalnum()]


def update_doc_score(new_score):
    """
    perform AND operation, to update score for documents
    :param new_score: score for documents in a query
    :return: updated score
    """
    return None


def query_phrase(query):
    return 0


def query_free_text(query_terms):
    global dictionary

    uniq_query_terms = list(set(query_terms))

    # construct query vector (tf * idf)
    # still no need to normalise the query vector
    query_counter = Counter(query_terms)
    query_vector = []
    # TODO: change it to follow the new format of dictionary
    for i in range(len(uniq_query_terms)):
        term = uniq_query_terms[i]
        query_tf = 1 + math.log(query_counter[term], 10)
        if term in dictionary['dictionary'].keys():
            query_idf = math.log(dictionary['collection_size'] / dictionary['dictionary'][term][0], 10)
        else:
            query_idf = 0
        query_weight = query_tf * query_idf
        query_vector.append(query_weight)

    pass


dictionary_file = postings_file = query_file = output_file_of_results = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-d':
        dictionary_file  = a
    elif o == '-p':
        postings_file = a
    elif o == '-q':
        query_file = a
    elif o == '-o':
        file_of_output = a
    else:
        assert False, "unhandled option"

if dictionary_file == None or postings_file == None or query_file == None or file_of_output == None :
    usage()
    sys.exit(2)

run_search(dictionary_file, postings_file, query_file, file_of_output)