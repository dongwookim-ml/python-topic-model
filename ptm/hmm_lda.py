import numpy as np

class HMM_LDA:
    """ implementation of HMM-LDA proposed by Griffiths et al. (2004)
     Original reference : Integrating topics and syntax, Griffiths, Thomas L and Steyvers, Mark and Blei, David M and Tenenbaum, Joshua B, NIPS 2004
    """

    def __init__(self, num_class, num_topic, num_voca, docs):
        self.C = num_class
        self.K = num_topic
        self.V = num_voca

        self.docs = docs
        self.N = len(docs)

        self.gamma = 0.1 
        self.eta = .5
        self.beta = 0.01
        self.alpha = 0.1

        self.CW = np.zeros([self.C,self.V]) + self.gamma    #class x word
        self.KW = np.zeros([self.K,self.V]) + self.beta     #topic x word
        self.DK = np.zeros([self.N,self.K]) + self.alpha    #document x topic
        self.T = np.zeros([self.C+2,self.C+2]) + self.eta   #class transition matix, including starting class(self.C) and end class(self.C+1)

    # randomly initialize 
    def random_init(self):
        self.word_class = list()
        self.word_topic = list()

        for di in xrange(self.N):
            doc = self.docs[di]
            num_sentence = len(doc)

            doc_class= list()
            doc_topic = list()

            for si in  xrange(num_sentence):
                sentence_class = list()
                sentence_topic = list()

                sentence = doc[si]
                len_sentence = len(sentence)

                for wi in xrange(len_sentence):
                    word = sentence[wi]
                    c = np.random.randint(self.C)

                    sentence_class.append(c)
                    self.CW[c,word] += 1
                    if wi == 0:     # if the first word
                        self.T[self.C,c] += 1
                    else:
                        self.T[sentence_class[wi-1],c] += 1

                    if wi == len_sentence-1:    #the last word
                        self.T[c,self.C+1] += 1

                    if c == 0:
                        k = np.random.randint(self.K)
                        sentence_topic.append(k)
                        self.DK[di, k] += 1
                        self.KW[k,word] += 1
                    else:
                        sentence_topic.append(-1)

                doc_class.append(sentence_class)
                doc_topic.append(sentence_topic)

            self.word_class.append(doc_class)
            self.word_topic.append(doc_topic)
        
    def gibbs_sampling(self, num_iter):

        for iter in xrange(num_iter):

            for di in xrange(self.N):
                doc = self.docs[di]
                num_sentence = len(doc)

                doc_topic = self.word_topic[di]
                doc_class = self.word_class[di]

                for si in  xrange(num_sentence):
                    sentence = doc[si]
                    len_sentence = len(sentence)

                    sentence_topic = doc_topic[si]
                    sentence_class = doc_class[si]

                    for wi in xrange(len_sentence):
                        word = sentence[wi]

                        if wi != 0:
                            prev_c = sentence_class[wi-1]
                        else:
                            prev_c = self.C

                        curr_c = sentence_class[wi]

                        if wi == len_sentence-1:
                            next_c = self.C+1
                        else:
                            next_c = sentence_class[wi+1]

                        #remove previous state
                        self.CW[curr_c,word] -= 1
                        self.T[prev_c,curr_c] -= 1
                        self.T[curr_c,next_c] -= 1

                        if curr_c == 0: #topic class
                            topic = sentence_topic[wi]
                            self.DK[di,topic] -= 1
                            self.KW[topic,word] -= 1


                        prob = np.zeros(self.K + self.C - 1)
                        #compute probability of being class c
                        for k in xrange(self.K):
                            prob[k] = np.float(self.T[prev_c,0])/np.sum(self.T[prev_c,:]) * self.T[0,next_c]/np.sum(self.T[0,:]) * self.KW[k,word]/np.sum(self.KW[k,:]) * self.DK[di,k]

                        for c in xrange(1, self.C):
                            prob[self.K+c-1] = np.float(self.T[prev_c,c])/np.sum(self.T[prev_c,:]) * self.T[c,next_c]/np.sum(self.T[c,:]) * self.CW[c,word]/np.sum(self.CW[c,:])

                        prob /= np.sum(prob)
                        sample = np.random.multinomial(1,prob).argmax()
                        if sample < self.K:
                            c = 0
                            self.DK[di,sample] += 1
                            self.KW[sample,word] += 1
                            sentence_topic[wi] = sample
                        else:
                            c = sample-self.K+1

                        sentence_class[wi] = c

                        self.CW[c,word] += 1
                        self.T[prev_c,c] += 1
                        self.T[c,next_c] += 1


def main():
    docs = [[[0,1,2,3],[2,3,4]], [[0,1,3],[3,4,5]], [[0,3,4],[4,5,6,6]]] #doc-sentence-word

    num_voca = 7
    num_class = 2
    num_topic = 3

    model = HMM_LDA(num_class,num_topic,num_voca, docs)
    model.random_init()
    model.gibbs_sampling(50)

    print model.T
    print np.sum(model.T)

if __name__ == '__main__':
    main()
