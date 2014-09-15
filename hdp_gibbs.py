import numpy as np
from collections import Counter
from scipy.special import gammaln
from sample_utils import sampling_from_dict

def log_normalize(val_list):
    maxval = max(val_list)
    new_val = [val-maxval for val in val_list]
    val_sum = sum(new_val)
    return [np.exp(val)/np.exp(val_sum) for val in new_val]


class WordTopicMatrix:
    def __init__(self, voca_size):
        self.topicSum = dict()
        self.wordTopic = dict()
        self.W = voca_size

    def increase(self, wordNo,topicNo,incVal = 1):
        if not self.topicSum.has_key(topicNo):
            self.wordTopic[topicNo] = np.zeros(self.W)
            self.topicSum[topicNo] = 0
        self.wordTopic[topicNo][wordNo] += incVal
        self.topicSum[topicNo] += incVal

    def decrease(self, wordNo,topicNo,decVal = 1):
        self.wordTopic[topicNo][wordNo] -= decVal
        self.topicSum[topicNo] -= decVal

        if self.topicSum[topicNo] == 0:
            del self.topicSum[topicNo]
            del self.wordTopic[topicNo]

    def get_topics(self):
        return self.topicSum.keys()

    def get_new_topic(self):
        new_topic = 0
        while True:
            if not self.topicSum.has_key(new_topic):
                return new_topic
            new_topic +=1

    def get_conditional(self, wordNo, topicNo, beta):
        if not self.topicSum.has_key(topicNo):
            return 1.0/self.W
        return (self.wordTopic[topicNo][wordNo]+beta)/(self.topicSum[topicNo] + self.W*beta)

    def get_multiword_log_conditional(self, word_list, topicNo, beta):
        counter = Counter(word_list)

        logval = gammaln(self.topicSum[topicNo] + beta*self.W) - gammaln(self.topicSum[topicNo] + len(word_list) + beta*self.W)

        for wordNo, value in counter.iteritems():
            logval += gammaln(beta + self.wordTopic[topicNo][wordNo] + value) - gammaln(beta + self.wordTopic[topicNo][wordNo])

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
        if self.tableSum.has_key(tableNo):
            return self.tableSum[tableNo]
        return 0

    def get_doc_length(self):
        return len(self.word_list)

    def get_tables(self):
        return self.tableSum.keys()

    def get_new_table(self):
        new_table = 0
        while True:
            if not self.tableSum.has_key(new_table):
                break
            new_table += 1
        return new_table

    def add_word_to_table(self, wordNo, tableNo):
        if not self.tableSum.has_key(tableNo):
            self.tableWords[tableNo] = dict()
            self.tableSum[tableNo] = 0

        self.tableSum[tableNo] += 1

        if not self.tableWords[tableNo].has_key(wordNo):
            self.tableWords[tableNo][wordNo] = 0

        self.tableWords[tableNo][wordNo] += 1

    def remove_word_from_table(self, wordNo, tableNo):
        self.tableWords[tableNo][wordNo] -= 1
        self.tableSum[tableNo] -= 1

        assert self.tableSum >= 0

        if self.tableWords[tableNo][wordNo] == 0:
            del self.tableWords[tableNo][wordNo]

        if self.tableSum[tableNo] == 0:
            del self.tableSum[tableNo]
            del self.tableWords[tableNo]

class HDP:

    def __init__(self, docs, voca_size, alpha = 1., alpha_H=1., beta = 0.1):
        self.alpha = alpha
        self.alpha_H = alpha_H
        self.beta = beta

        self.doc_list = list()
        self.word_topic = WordTopicMatrix(voca_size)

        self.table_assigned_topics = dict()
        self.total_table = 0.

        for doc in docs:
            document = Document()
            for wordNo in doc:
                word = Word(wordNo)
                document.word_list.append(word)
            self.doc_list.append(document)

    def gibbs_sampling(self, max_iter):
        for iteration in xrange(max_iter):
            self.sampling_tables(iteration)
            self.sampling_dishes(iteration)

    def sampling_dishes(self, iteration):
        # need to check this function
        
        for doc in self.doc_list:
            tables = doc.get_tables()

            for table in tables:
                tableWords = doc.tableWords[table]
                old_topic = doc.tableTopic[table]

                #remove current table
                self.table_assigned_topics[old_topic] -= 1
                if self.table_assigned_topics[old_topic] == 0:
                    del self.table_assigned_topics[old_topic]
                for wordNo, counts in tableWords.iteritems():
                    self.word_topic.decrease(wordNo, old_topic, counts)

                topic_prob = dict()
                for topicNo in self.table_assigned_topics.keys():
                    topic_prob[topicNo] = np.log(self.table_assigned_topics[topicNo]) + self.word_topic.get_multiword_log_conditional(tableWords, topicNo, self.beta)
                    
                new_topic_no = self.get_new_topic()

                topic_prob[new_topic_no] = np.log(self.alpha_H) + self.word_topic.get_multiword_log_conditional(tableWords, new_topic_no, self.beta)

                topic_prob = log_normalize(topic_prob)
                new_topic = sampling_from_dict(topic_prob)

                #add current table
                doc.tableTopic[table] = new_topic
                if new_topic == new_topic_no:
                    self.table_assigned_topics[new_topic] = 0
                self.table_assigned_topics[new_topic] += 1
                for wordNo, counts in tableWords.iteritems():
                    self.word_topic.increase(wordNo, new_topic, counts)
                
    def sampling_tables(self, iteration):

        for doc in self.doc_list:
            doc_length = doc.get_doc_length() - 1

            for word in doc.word_list:
                wordNo = word.wordNo

                #remove current word
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

                #compute conditional for each table, topic
                tables = doc.get_tables()
                topic_prob = dict()
                for topicNo in self.word_topic.get_topics():
                    topic_prob[topicNo] = self.word_topic.get_conditional(wordNo,topicNo,self.beta)

                new_topic_no = self.word_topic.get_new_topic()
                topic_prob[new_topic_no] = self.word_topic.get_conditional(wordNo,new_topic_no, self.beta)

                table_prob = dict()
                for tableNo in tables:
                    table_prob[tableNo] = topic_prob[doc.tableTopic[tableNo]] * (doc.tableSum[tableNo])/(doc_length + self.alpha)

                new_table_no = doc.get_new_table()
                new_table_prob = 0
                new_table_dict = dict()
                for topicNo in topic_prob.keys():
                    if self.table_assigned_topics.has_key(topicNo):
                        prob = (self.table_assigned_topics[topicNo])/(self.total_table + self.alpha_H) * topic_prob[topicNo]
                    else:
                        prob = self.alpha_H/(self.total_table + self.alpha_H) * topic_prob[topicNo]
                    new_table_prob += prob
                    new_table_dict[topicNo] = prob

                table_prob[new_table_no] = new_table_prob * self.alpha / (doc_length + self.alpha)

                new_table = sampling_from_dict(table_prob)

                if new_table == new_table_no:
                    new_topic_of_new_table = sampling_from_dict(new_table_dict)
                    self.total_table += 1
                    if new_topic_of_new_table == new_topic_no: 
                        self.table_assigned_topics[new_topic_of_new_table] = 0
                    self.table_assigned_topics[new_topic_of_new_table] += 1
                    doc.tableTopic[new_table] = new_topic_of_new_table

                word.tableNo = new_table
                doc.add_word_to_table(wordNo, new_table)
                self.word_topic.increase(wordNo, doc.tableTopic[new_table])

if __name__ == '__main__':
    #test code
    docs = [[0,1,2,3,3,4,3,4,5], [2,3,3,5,6,7,8,3,8,9,5]]
    model = HDP(docs, 10)
    model.gibbs_sampling(10)
    print model.word_topic.wordTopic

