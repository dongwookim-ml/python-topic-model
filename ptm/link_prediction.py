import cPickle
#cd Desktop/git/author_citation/cora

doc_cnt = cPickle.load(open('doc_cnt.pkl'))
doc_ids = cPickle.load(open('doc_ids.pkl'))
doc_links = cPickle.load(open('doc_links_sym.pkl'))
bow = cPickle.load(open('bow.pkl'))

'''
test_docs = random.choice(range(len(doc_cnt)), size = 100, replace = False)
test_list = list(test_docs)
for v in test_docs:
	test_list += doc_links[v]
'''

test_docs = random.choice(range(len(doc_cnt)), size = 100, replace = False)

#for v in test_links:
for v in test_docs:	
	for v2 in doc_links[v]:
		try:
			doc_links[v2].remove(v)
		except:
			continue
	doc_links[v] = []

doc_links_unremoved = cPickle.load(open('doc_links_sym.pkl'))

#model = rtm(30, len(doc_cnt), len(bow), doc_ids, doc_cnt, doc_links, rho = 3000)
#model.posterior_inference(50)

for rho in [0, 10, 10,  100, 100, 1000, 1000, 10000, 10000]:
#for rho in [10000, 100000, 1000000]:
    model = rtm(30, len(doc_cnt), len(bow), doc_ids, doc_cnt, doc_links, rho = rho)
    model.posterior_inference(30)
    print

