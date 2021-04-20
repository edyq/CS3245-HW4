#!/usr/bin/python3
import sys
import math
import getopt
import pickle
import string
import heapq
import re
import nltk
from nltk import PorterStemmer
from nltk import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from collections import defaultdict
from collections import Counter

dictionary = {}
collection_size = 0
nltk.download('stopwords')
nltk.download('wordnet')  # need to ask prof whether these already exist on server
stop_words=set(stopwords.words("english"))


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

    # step 1: treat everything as free text query; obtain doc_score
    query_line = open(query_file, 'r').readline().rstrip()
    free_text_query = process_query_as_free_text(query_line)
    expanded_query_tokens = tokenize_free_text(free_text_query)
    doc_score = query_free_text(expanded_query_tokens)

    # step 2: extract the phrasal query; obtain no_matched_phrases / no_phrases_in_the_query
    phrasal_query = extract_phrasal_query(query_line)
    phrasal_doc_score = query_phrase(phrasal_query)

    # step 3: adjust doc_score for those with matching phrases
    adjusted_doc_score = adjust_doc_score(doc_score, phrasal_doc_score, 0.1)

    # step 4: sort and write to results_file
    result = ' '.join(map(str, [k for k, v in sorted(adjusted_doc_score.items(), key=lambda item: item[1], reverse=True)]))
    out_file = open(results_file, 'w')
    out_file.write(result)
    out_file.close()


def expand_query(tokens):
    """
    Expand the original free text query with synonyms from wordnet
    :param tokens: ['token_1', 'token_2']
    :return: a list of synonyms
    """
    synonyms = []
    count = 0
    for term in tokens:
        for syn in wordnet.synsets(term):
            if count >= 3:
                count = 0
                break
            else:
                for l in syn.lemmas():
                    if count >= 3:
                        break
                    else:
                        if l.name() not in synonyms:
                            synonyms.append(l.name())
                            count += 1
    return synonyms


def extract_phrasal_query(query):
    """
    Extract the phrasal queries from the original query with regex
    :param query: string
    :return: list
    """
    phrases = re.findall(r'\"(.+?)\"', query)
    stemmer = PorterStemmer()
    result = []
    for phrase in phrases:
        result.append([stemmer.stem(word) for word in word_tokenize(phrase.lower())])
    return result


def process_query_as_free_text(query):
    """
    FOR NOW: doing everything as free text; adjust score for doc matching the phrase
    :param query: string
    :return: string

    """
    query_list = query.split(' AND ')
    processed_query = []
    for query in query_list:
        for word in query:
            if word[0] == '"':
                word = word[1:]
            elif word[-1] == '"':
                word = word[:-1]
            processed_query.append(word)

    return ''.join(processed_query)


def tokenize_free_text(query):
    """
    1. lower case
    2. word_tokenize
    3. query expansion
    :param query:
    :return:
    """
    tokens = word_tokenize(query.lower())
    stemmer = PorterStemmer()

    filtered_tokens = []
    for word in tokens:
        if word not in string.punctuation and word not in stop_words:
            filtered_tokens.append(word)

    print("original query: ", filtered_tokens)

    # query expansion using wordnet
    expanded_tokens = expand_query(filtered_tokens)
    print("expanded query: ", expanded_tokens)
    return [stemmer.stem(word) for word in expanded_tokens]


def adjust_doc_score(doc_score, phrasal_doc_score, alpha):
    """
    adjust score for docs that match the exact phrasal query
    :param new_score: score for documents in a query
    :return: updated score
    """
    score_average = sum(doc_score.values()) / len(doc_score)
    for doc in doc_score.keys():
        if doc in phrasal_doc_score.keys():
            doc_score[doc] += score_average * alpha * phrasal_doc_score[doc]
    return doc_score


def query_phrase(query):
    """
    still using lnt.ltc
    :param query = [['phrasal', 'query'], ['phrasal', 'query']]
    :return: phrasal_doc_score = {doc_id: match_percentage, doc_id: match_percentage}
    """
    global dictionary
    global collection_size

    no_phrases = len(query)
    phrasal_doc_score = defaultdict(lambda: 0)
    phrase_postings = []
    p_file = open(postings_file, 'rb')
    for phrase in query:
        phrase_postings.append([])
        for term in phrase:
            if term in dictionary.keys():
                offset = dictionary[term][1]
                p_file.seek(offset)
                posting = pickle.load(p_file)
                phrase_postings[-1].append(posting)
            else:
                print('unknown query term ' + term)
                sys.exit(2)
    p_file.close()

    for i in range(no_phrases):
        valid_docs = []
        if len(query[i]) == 2:
            valid_docs = query_biword(phrase_postings[i])
        elif len(query[i]) == 3:
            valid_docs = query_triword(phrase_postings[i])
        else:
            print('only phrases with 2 or 3 words are supported')
            sys.exit(2)
        for doc in valid_docs:
            phrasal_doc_score[doc] += 1 / no_phrases

    return phrasal_doc_score


def query_biword(phrase_postings):
    """
    return docs matching the phrase
    :param phrase: ['phrasal', 'query']
    :param phrase_postings: [{doc_id: [tf-idf, [position index, position index]], doc_id:[]}, {doc_id:[], }]
    :return: list of doc ids that matches the phrase
    """
    result = []
    doc_set_1 = set(list(map(str, phrase_postings[0].keys())))
    doc_set_2 = set(list(map(str, phrase_postings[1].keys())))
    docs = list(doc_set_1.intersection(doc_set_2))

    for doc in docs:
        ptr_1 = 0
        ptr_2 = 0
        posting_1 = phrase_postings[0][doc][1]
        posting_2 = phrase_postings[1][doc][1]
        while ptr_1 < len(posting_1) and ptr_2 < len(posting_2):
            if posting_1[ptr_1] + 1 == posting_2[ptr_2]:
                result.append(doc)
                break
            elif posting_1[ptr_1] + 1 > posting_2[ptr_2]:
                ptr_2 += 1
            else:
                ptr_1 += 1
    print(result)
    return result


def query_triword(phrase_postings):
    """
    :param phrase:
    :param phrase_postings:
    :return:
    """
    result = []
    doc_set_1 = set(list(map(str, phrase_postings[0].keys())))
    doc_set_2 = set(list(map(str, phrase_postings[1].keys())))
    doc_set_3 = set(list(map(str, phrase_postings[2].keys())))
    docs = list(doc_set_1.intersection(doc_set_2).intersection(doc_set_3))

    for doc in docs:
        ptr_1 = 0
        ptr_2 = 0
        ptr_3 = 0
        posting_1 = phrase_postings[0][doc][1]
        posting_2 = phrase_postings[1][doc][1]
        posting_3 = phrase_postings[2][doc][1]
        while ptr_1 < len(posting_1) and ptr_2 < len(posting_2) and ptr_3 < len(posting_3):
            if posting_1[ptr_1] + 1 == posting_2[ptr_2] and posting_2[ptr_2] + 1 == posting_3[ptr_3]:
                result.append(doc)
                break
            elif posting_1[ptr_1] + 1 > posting_2[ptr_2]:
                ptr_2 += 1
            elif posting_2[ptr_2] + 1 > posting_3[ptr_3]:
                ptr_3 += 1
            else:
                ptr_1 += 1
    print(result)
    return result


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