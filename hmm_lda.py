import numpy as np

class HMM_LDA:
    def __init__(self, num_class, num_topic, num_voca, docs):
        self.C = num_class
        self.K = num_topic
        self.V = num_voca

        self.docs = docs
        self.N = len(docs)

        self.gamma = 1  
        self.eta = 1
        self.beta = 0.01
        self.alpha = 0.1

        self.CW = np.zeros([self.C,self.V]) + self.gamma
        self.KW = np.zeros([self.K,self.V]) + self.beta
        self.DK = np.zeros([self.N,self.K]) + self.alpha
        self.T = np.zeros([self.C+2,self.C+2]) + self.eta #transition matix, including starting class and end class


    def gibbs_sampling(self, num_iter):

        for iter in xrange(num_iter):

            for doc in docs:
                num_sentence = len(doc)

                for si in  xrange(num_sentence):
                    sentence = doc[si]
                    len_sentence = len(sentence)

                    for wi in xrange(len_sentence):
                        word = sentence[wi]

                        prob = np.zeros(self.C)
                        #compute probability of being class c
                        #starting word
                        if wi == 0: 

                        elif wi == num_sentence-1:

                        else:

                        #compute probability of being topic k if class c = topic class


def main():
    docs = [[[0,1,2,3],[2,3,4]], [[0,1,3],[3,4,5]], [[0,3,4],[4,5,6,6]]] #doc-sentence-word

    num_voca = 7
    num_class = 2
    num_topic = 3

    model = HMM_LDA(num_class,num_topic,num_voca, docs)
    model.gibbs_sampling(10)

if __name__ == '__main__':
    main()
