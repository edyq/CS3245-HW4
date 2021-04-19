#!/usr/bin/python3
import sys
import math
import getopt
import pickle
import string
import heapq
from nltk import PorterStemmer
from nltk import word_tokenize
from collections import defaultdict
from collections import Counter

dictionary = {}
collection_size = 0


def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q query-file -o output-file-of-results")


def run_search(dict_file, postings_file, query_file, results_file):
    print('running search on the query...')

    global dictionary
    global collection_size
    dic_file = open(dict_file, 'rb')
    dictionary_obj = pickle.load(dic_file)
    dictionary = dictionary_obj['dictionary']
    collection_size = dictionary_obj['collection_size']
    dic_file.close()

    # TODO: make use of relevant docs provided in the query
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

    # TODO: handle query like '"small dog" puppy'
    # https://piazza.com/class/kjmny91pkrx6ag?cid=214
    # phrasal and Boolean queries shouldn't be treated as strict operators, but rather influence the ranking of
    # documents retrieved. That is, you likely want to return "little puppy chihuahua" documents at the top, but also
    # still return "little chihuahua puppy" somewhere below.

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
    """
    1. lower case
    2. word_tokenize
    3.
    :param query:
    :return:
    """
    tokens = word_tokenize(query.lower())
    stemmer = PorterStemmer()
    cleaned_field = []

    for word in tokens:
        if word not in string.punctuation:
            # hopefully no such thing appears in search
            # word = remove_prefix_num(word)
            # word = remove_attached_punctuation(word)
            word = stemmer.stem(word)
            if word not in string.punctuation:
                cleaned_field.append(word)
    return cleaned_field


def update_doc_score(new_score):
    """
    perform AND operation, to update score for documents
    :param new_score: score for documents in a query
    :return: updated score
    """
    return None


def query_phrase(query):
    """
    still using lnt.ltc
    :param query:
    :return:
    """
    global dictionary
    global collection_size
    # TODO ??????????? What's this supposed to mean ???????????
    # https://piazza.com/class/kjmny91pkrx6ag?cid=214
    # phrasal and Boolean queries shouldn't be treated as strict operators, but rather influence the ranking of
    # documents retrieved. That is, you likely want to return "little puppy chihuahua" documents at the top, but also
    # still return "little chihuahua puppy" somewhere below.

    # idf for query phrase
    # tf = 1 since it is a phrasal query and I treat a phrase as 1 term
    # TODO - calculate idf for the query phrase
    phrase_postings = []
    p_file = open(postings_file, 'rb')
    for term in query:
        offset = dictionary[term][1]
        p_file.seek(offset)
        posting = pickle.load(p_file)
        phrase_postings.append(posting)
    p_file.close()


    # tf for query phrase in the document
    # TODO - tf for the phase in the documents


def query_free_text(query_terms):
    """
    still using lnc.ltc
    :param query_terms:
    :return:
    """
    global dictionary
    global collection_size
    uniq_query_terms = list(set(query_terms))

    # construct query vector (tf * idf)
    # still no need to normalise the query vector
    query_counter = Counter(query_terms)
    query_vector = []

    for i in range(len(uniq_query_terms)):
        term = uniq_query_terms[i]
        query_tf = 1 + math.log(query_counter[term], 10)
        if term in dictionary.keys():
            query_idf = math.log(collection_size / dictionary[term][0], 10)
        else:
            query_idf = 0
        query_weight = query_tf * query_idf
        query_vector.append(query_weight)

    # retrieve normalised doc tf (precalculated in indexing phrase)
    document_vector = defaultdict(lambda: [0.] * len(uniq_query_terms))
    for i in range(len(uniq_query_terms)):
        term = uniq_query_terms[i]
        if term in dictionary.keys():
            offset = dictionary[term][1]
            p_file = open(postings_file, 'rb')
            p_file.seek(offset)
            posting = pickle.load(p_file)
            p_file.close()
            for doc in posting.keys():
                document_vector[doc][i] = posting[doc][0]

    # calculate score for each doc
    doc_score = {}
    for doc, vector in document_vector.items():
        doc_score[doc] = sum([i*j for (i, j) in zip(vector, query_vector)])
    return doc_score


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