python-topic-model
==================

Implementations of various topic models written in Python. Note that some of the implementations (the models with MCMC) are extremely slow. I do not recommend to use it for large scale datasets.

Current implementations
-----------------------

* Latent Dirichlet allocation
  * [Collapsed Gibbs sampling](http://nbviewer.jupyter.org/github/arongdari/python-topic-model/blob/master/notebook/LDA_example.ipynb)
  * [Variational inference](http://nbviewer.jupyter.org/github/arongdari/python-topic-model/blob/master/notebook/LDA_example.ipynb)
* Collaborative topic model
  * Variational inference
* Relational topic model (VI)
  * [Exponential link function](http://nbviewer.jupyter.org/github/arongdari/python-topic-model/blob/master/notebook/RelationalTopicModel_example.ipynb)
* [Author-Topic model](http://nbviewer.jupyter.org/github/arongdari/python-topic-model/blob/master/notebook/AuthorTopicModel_example.ipynb)
* [HMM-LDA](http://nbviewer.jupyter.org/github/arongdari/python-topic-model/blob/master/notebook/HMM_LDA_example.ipynb)
* Discrete infinite logistic normal (DILN)
  * Variational inference
* Supervised topic model
  * [Stochastic (Gibbs) EM](http://nbviewer.jupyter.org/github/arongdari/python-topic-model/blob/master/notebook/SupervisedTopicModel_example.ipynb)
  * Variational inference
* Hierarchical Dirichlet process
  * Collapsed Gibbs sampling
* Hierarchical Dirichlet scaling process

