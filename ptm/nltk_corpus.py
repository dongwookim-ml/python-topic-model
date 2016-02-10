from collections import Counter

import numpy as np
import nltk
from nltk import word_tokenize
from nltk.corpus import reuters, stopwords
from six.moves import xrange


def get_ids_cnt(corpus, max_voca=9999999, remove_top_n=5):
    voca = set(w.lower() for w in nltk.corpus.words.words())
    stop = stopwords.words('english')

    docs = list()
    freq = Counter()

    for doc in corpus:
        if isinstance(doc, str):
            doc = word_tokenize(doc)
        elif not hasattr(doc, '__iter__'):
            raise Exception('Corpus is not a list of string or token list')

        # remove word using stopword list or single character word
        doc = [word.lower() for word in doc if word.lower() in voca and word.lower() not in stop and len(word) != 1]
        freq.update(doc)
        docs.append(doc)

    voca = [key for iter, (key, val) in enumerate(freq.most_common(max_voca)) if iter >= remove_top_n]

    voca_dic = dict()
    voca_list = list()
    for word in voca:
        voca_dic[word] = len(voca_dic)
        voca_list.append(word)

    doc_ids = list()
    doc_cnt = list()

    for doc in docs:
        words = set(doc)
        ids = np.array([int(voca_dic[word]) for word in words if word in voca_dic])
        cnt = np.array([int(doc.count(word)) for word in words if word in voca_dic])

        doc_ids.append(ids)
        doc_cnt.append(cnt)

    return np.array(voca_list), doc_ids, doc_cnt


def get_reuters_cnt_ids(num_doc=100, max_voca=10000, remove_top_n=5):
    """To get test data for training a model
    reuters, stopwords, english words corpora should be installed in nltk_data: nltk.download()

    Parameters
    ----------
    num_doc: int
        number of documents to be returned
    max_voca: int
        maximum number of vocabulary size for the returned corpus
    remove_top_n: int
        remove top n frequently used words

    Returns
    -------
    voca_list: ndarray
        list of vocabulary used to construct a corpus
    doc_ids: list
        list of list of word id for each document
    doc_cnt: list
        list of list of word count for each document
    """
    file_list = reuters.fileids()
    corpus = [reuters.words(file_list[i]) for i in xrange(num_doc)]

    return get_ids_cnt(corpus, max_voca, remove_top_n)
