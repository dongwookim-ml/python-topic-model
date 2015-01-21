import numpy as np

def isfloat(value):
    """
    Check the value is convertable to float value
    """
    try:
        float(value)
        return True
    except ValueError:
        return False

def read_voca(path):
    """
    open file from path and read each line to return the word list
    """
    with open(path, 'r') as f:
        return [word.strip() for word in f.readlines()]

def word_cnt_to_bow_list(word_ids,word_cnt):
    corpus_list = list()
    for di in xrange(len(word_ids)):
        doc_list = list()
        for wi in xrange(len(word_ids[di])):
            word = word_ids[di][wi]
            for c in xrange(word_cnt[di][wi]):
                doc_list.append(word)
        corpus_list.append(doc_list)
    return corpus_list

def get_f1(truth, predict):
    """
    truth and predict are arrays in which values are True or False
    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    truth = truth[:]
    predict = predict[:]

    tp = predict[truth==True].sum()
    fp = predict[truth==False].sum()
    fn = len(predict[truth==True]) - tp
    tn = len(predict[truth==False]) - fp

    if tp + fp != 0:
        precision = np.float(tp)/np.float(tp + fp)
    else:
        precision = 0
    recall = np.float(tp)/np.float(tp + fn)

    if precision == 0 and recall == 0:
        return 0, 0, 0

    f1 = 2.*precision*recall / (precision + recall)
    return f1, precision, recall

def get_f1_from_confusion(confusion_matrix):
    ndim = confusion_matrix.shape[0]

    micro_f1 = 0.
    macro_f1 = 0.

    cnt = 0.
    m_tp=0.
    m_fp=0.
    m_fn=0.

    avg_prc = 0
    avg_rcl = 0

    f1_list = list()
    prc_list = list()
    rcl_list = list()

    for dim in xrange(ndim):
        tp = confusion_matrix[dim,dim]
        fp = np.sum(confusion_matrix[:,dim]) - tp
        fn = np.sum(confusion_matrix[dim,:]) - tp
        m_tp += tp
        m_fp += fp
        m_fn += fn

        if tp+fp+fn != 0:
            micro_f1 += 2.*tp/(2.*tp + fp + fn)
            cnt += 1.
            f1_list.append(2.*tp/(2.*tp + fp + fn))
        else:
            f1_list.append(0)

        if tp+fp != 0:
            avg_prc += (tp/(tp+fp))
            prc_list.append(tp/(tp+fp))
        else:
            prc_list.append(0)

        if tp+fn != 0:
            avg_rcl += (tp/(tp+fn))
            rcl_list.append(tp/(tp+fn))
        else:
            rcl_list.append(0)

    micro_f1 = (micro_f1/cnt)
    macro_f1 = 2.*m_tp/(2.*m_tp+m_fp+m_fn)
    avg_prc /= cnt
    avg_rcl /= cnt

    return micro_f1, macro_f1, f1_list, prc_list, rcl_list

def get_mse_from_confusion(confusion_matrix, ratings):
    ndim = confusion_matrix.shape[0]

    mse = 0
    for dim in xrange(ndim):
        for dim2 in xrange(ndim):
            mse += (np.abs(ratings[dim] - ratings[dim2]))*confusion_matrix[dim,dim2]

    return mse

def log_normalize(log_v_vector):
    """
    returns a probability vector of the log probability vector
    """
    max_v = log_v_vector.max()
    log_v_vector += max_v
    log_v_vector = np.exp(log_v_vector)
    log_v_vector /= log_v_vector.sum()
    return log_v_vector


def convert_wrdcnt_wrdlist(corpus_ids, corpus_cnt):
    corpus = list()

    for di in xrange(len(corpus_ids)):
        doc = list()
        doc_ids = corpus_ids[di]
        doc_cnt = corpus_cnt[di]
        for wi in xrange(len(doc_ids)):
            word_id = doc_ids[wi]
            for si in xrange(doc_cnt[wi]):
                doc.append(word_id)
        corpus.append(doc)
    return corpus
    

def write_top_words(topic_word_matrix, vocab, filepath, num_top_words = 20, delimiter=',', newline='\n'):
    with open(filepath, 'w') as f:
        for ti in xrange(topic_word_matrix.shape[0]):
            top_words = vocab[topic_word_matrix[:,ti].argsort()[::-1][:num_top_words]]
            f.write( '%d' % (ti) )
            for word in top_words:
                f.write(delimiter + word)
            f.write(newline)
