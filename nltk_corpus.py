import numpy as np
from nltk.corpus import reuters
from collections import Counter


def get_reuters_cnt_ids(num_doc=100, max_voca=10000):
    ''' to get test data for training a model
    reuters corpus should be installed in nltk_data: nltk.download()
    '''
    file_list = reuters.fileids()
    
    docs = list()
    freq = Counter()

    for i in range(num_doc):
        doc = reuters.words(file_list[i])
        freq.update(doc)
        docs.append(doc)

    voca = [key for key,val in freq.most_common(max_voca)]

    voca_dic = dict()
    voca_list = list()
    for word in voca:
        voca_dic[word] = len(voca_dic)
        voca_list.append(word)

    doc_ids = list()
    doc_cnt = list()

    for doc in docs:
        words = set(doc)
        ids = np.array([int(voca_dic[word]) for word in words if voca_dic.has_key(word)])
        cnt = np.array([int(doc.count(word)) for word in words if voca_dic.has_key(word)])

        doc_ids.append(ids)
        doc_cnt.append(cnt)

    return np.array(voca_list), doc_ids, doc_cnt

