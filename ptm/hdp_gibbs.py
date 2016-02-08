import numpy as np
from collections import Counter
from scipy.special import gammaln

from .utils import sampling_from_dict

# normalize the log valued list
def log_normalize(val_list):
    maxval = max(val_list)
    new_val = [val - maxval for val in val_list]
    val_sum = sum(new_val)
    return [np.exp(val) / np.exp(val_sum) for val in new_val]


class WordTopicMatrix:
    def __init__(self, n_voca):
        self.topicSum = dict()
        self.wordTopic = dict()
        self.n_voca = n_voca

    def increase(self, wordNo, topicNo, incVal=1):
        """ increase a number of assigned word by incVal
        """
        if not self.topicSum.has_key(topicNo):
            self.wordTopic[topicNo] = np.zeros(self.n_voca)
            self.topicSum[topicNo] = 0
        self.wordTopic[topicNo][wordNo] += incVal
        self.topicSum[topicNo] += incVal

    def decrease(self, wordNo, topicNo, decVal=1):
        self.wordTopic[topicNo][wordNo] -= decVal
        self.topicSum[topicNo] -= decVal

        if self.topicSum[topicNo] == 0:
            del self.topicSum[topicNo]
            del self.wordTopic[topicNo]

    def get_topics(self):
        """ return a list of allocated topics 
        """
        return self.topicSum.keys()

    def get_new_topic(self):
        """ return a candidate topic number for a new topic
        """
        new_topic = 0
        while True:
            if not self.topicSum.has_key(new_topic):
                return new_topic
            new_topic += 1

    def get_conditional(self, wordNo, topicNo, eta):
        """ compute a marginalized probability (predictive distribution) of a word given a topic 
        """
        if not self.topicSum.has_key(topicNo):
            return 1.0 / self.n_voca
        return (self.wordTopic[topicNo][wordNo] + eta) / (self.topicSum[topicNo] + self.n_voca * eta)

    def get_multiword_log_conditional(self, word_list, topicNo, eta):
        """ compute a marginalized probability of a set of words given a topic
        """
        counter = Counter(word_list)

        logval = gammaln(self.topicSum[topicNo] + eta * self.n_voca) - gammaln(
            self.topicSum[topicNo] + len(word_list) + eta * self.n_voca)

        for wordNo, value in counter.iteritems():
            logval += gammaln(eta + self.wordTopic[topicNo][wordNo] + value) - gammaln(
                eta + self.wordTopic[topicNo][wordNo])

        return logval


class Word:
    def __init__(self, wordNo):
        self.wordNo = wordNo
        self.tableNo = 0


class Document:
    def __init__(self):
        self.tableSum = dict()
        self.tableTopic = dict()
        self.tableWords = dict()
        self.word_list = list()

    def get_table_sum(self, tableNo):
        """ return a number of assigned tokens to table tableNo
        """
        if self.tableSum.has_key(tableNo):
            return self.tableSum[tableNo]
        return 0

    def get_doc_length(self):
        """ return a length of a document (number of tokens)
        """
        return len(self.word_list)

    def get_tables(self):
        """ return a set of allocated tables
        """
        return self.tableSum.keys()

    def get_new_table(self):
        """ return a cadidate table number for a new table
        """
        new_table = 0
        while True:
            if not self.tableSum.has_key(new_table):
                break
            new_table += 1
        return new_table

    def add_word_to_table(self, wordNo, tableNo):
        """ assign a word to the desginated table
        """
        if not self.tableSum.has_key(tableNo):
            self.tableWords[tableNo] = dict()
            self.tableSum[tableNo] = 0

        self.tableSum[tableNo] += 1

        if not self.tableWords[tableNo].has_key(wordNo):
            self.tableWords[tableNo][wordNo] = 0

        self.tableWords[tableNo][wordNo] += 1

    def remove_word_from_table(self, wordNo, tableNo):
        """ remove a word from a table
        """
        self.tableWords[tableNo][wordNo] -= 1
        self.tableSum[tableNo] -= 1

        assert self.tableSum >= 0

        if self.tableWords[tableNo][wordNo] == 0:
            del self.tableWords[tableNo][wordNo]

        if self.tableSum[tableNo] == 0:
            del self.tableSum[tableNo]
            del self.tableWords[tableNo]


class HDP:
    """ Hierarchical Dirichlet process (HDP) with collapsed gibbs sampling algorithm for the posterior inference.
    """

    def __init__(self, docs, n_voca, beta=1., alpha=1., eta=0.1):
        """ follows the notation of Teh et al. (2006)

        Keyword arguments:
        docs = list of list of tokens [[][]]
        voca_size = size of vocabulary
        beta = second level concentration parameter (default 1.)
        alpha = first level concentration parameter (default 1.)
        eta = prior for word-topic (Dir) distribution (default .1)
        """
        self.beta = beta
        self.alpha = alpha
        self.eta = eta

        self.doc_list = list()
        self.word_topic = WordTopicMatrix(n_voca)

        self.table_assigned_topics = dict()
        self.total_table = 0.

        for doc in docs:
            document = Document()
            for wordNo in doc:
                word = Word(wordNo)
                document.word_list.append(word)
            self.doc_list.append(document)

    def gibbs_sampling(self, max_iter=100):
        """ posterior sampling for HDP

        Keyword arguments:
        max_iter -- a maximum number of iteration (default 100)
        """

        for iteration in xrange(max_iter):
            self.sampling_tables(iteration)
            self.sampling_dishes(iteration)

    def sampling_dishes(self, iteration):
        """ sample a topic of each table
        """
        # need to check this function

        for doc in self.doc_list:
            tables = doc.get_tables()

            for table in tables:
                tableWords = doc.tableWords[table]
                old_topic = doc.tableTopic[table]

                # remove current topic of table
                self.table_assigned_topics[old_topic] -= 1
                if self.table_assigned_topics[old_topic] == 0:
                    del self.table_assigned_topics[old_topic]
                for wordNo, counts in tableWords.iteritems():
                    self.word_topic.decrease(wordNo, old_topic, counts)

                topic_prob = dict()
                for topicNo in self.table_assigned_topics.keys():
                    topic_prob[topicNo] = np.log(
                        self.table_assigned_topics[topicNo]) + self.word_topic.get_multiword_log_conditional(tableWords,
                                                                                                             topicNo,
                                                                                                             self.eta)

                new_topic_no = self.get_new_topic()

                topic_prob[new_topic_no] = np.log(self.alpha) + self.word_topic.get_multiword_log_conditional(
                    tableWords, new_topic_no, self.eta)

                topic_prob = log_normalize(topic_prob)
                new_topic = sampling_from_dict(topic_prob)

                doc.tableTopic[table] = new_topic
                # if a new topic is chosen
                if new_topic == new_topic_no:
                    self.table_assigned_topics[new_topic] = 0
                self.table_assigned_topics[new_topic] += 1
                for wordNo, counts in tableWords.iteritems():
                    self.word_topic.increase(wordNo, new_topic, counts)

    def sampling_tables(self, iteration):
        """ iterate a corpus and sample a table of each word token

        Keyword arguments:
        iteration -- current iteration count
        """
        for doc in self.doc_list:
            doc_length = doc.get_doc_length() - 1

            for word in doc.word_list:
                wordNo = word.wordNo

                # remove current word from assigned table
                if iteration != 0:
                    old_table = word.tableNo
                    old_topic = doc.tableTopic[old_table]

                    doc.remove_word_from_table(wordNo, old_table)
                    if doc.get_table_sum(old_table) == 0:
                        self.table_assigned_topics[old_topic] -= 1
                        self.total_table -= 1
                        if self.table_assigned_topics[old_topic] == 0:
                            del self.table_assigned_topics[old_topic]
                    self.word_topic.decrease(wordNo, old_topic)

                # compute conditional for each table, topic
                tables = doc.get_tables()
                topic_prob = dict()
                for topicNo in self.word_topic.get_topics():
                    topic_prob[topicNo] = self.word_topic.get_conditional(wordNo, topicNo, self.eta)

                new_topic_no = self.word_topic.get_new_topic()
                topic_prob[new_topic_no] = self.word_topic.get_conditional(wordNo, new_topic_no, self.eta)

                table_prob = dict()
                for tableNo in tables:
                    table_prob[tableNo] = topic_prob[doc.tableTopic[tableNo]] * (doc.tableSum[tableNo]) / (
                        doc_length + self.beta)

                new_table_no = doc.get_new_table()
                new_table_prob = 0
                new_table_dict = dict()
                for topicNo in topic_prob.keys():
                    if self.table_assigned_topics.has_key(topicNo):
                        prob = (self.table_assigned_topics[topicNo]) / (self.total_table + self.alpha) * topic_prob[
                            topicNo]
                    else:
                        prob = self.alpha / (self.total_table + self.alpha) * topic_prob[topicNo]
                    new_table_prob += prob
                    new_table_dict[topicNo] = prob

                table_prob[new_table_no] = new_table_prob * self.beta / (doc_length + self.beta)

                new_table = sampling_from_dict(table_prob)

                # if a new table is chosen
                if new_table == new_table_no:
                    new_topic_of_new_table = sampling_from_dict(new_table_dict)
                    self.total_table += 1

                    # if a new topic is chosen for the new table
                    if new_topic_of_new_table == new_topic_no:
                        self.table_assigned_topics[new_topic_of_new_table] = 0
                    self.table_assigned_topics[new_topic_of_new_table] += 1
                    doc.tableTopic[new_table] = new_topic_of_new_table

                word.tableNo = new_table
                doc.add_word_to_table(wordNo, new_table)
                self.word_topic.increase(wordNo, doc.tableTopic[new_table])


if __name__ == '__main__':
    # test code
    docs = [[0, 1, 2, 3, 3, 4, 3, 4, 5], [2, 3, 3, 5, 6, 7, 8, 3, 8, 9, 5]]
    model = HDP(docs, 10)
    model.gibbs_sampling(10)
    print(model.word_topic.wordTopic)
