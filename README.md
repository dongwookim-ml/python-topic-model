python-topic-model
==================

Implementations of various topic models written in Python. Note that some of the implementations (the models with MCMC) are extremely slow. I do not recommend to use it for large scale datasets.

Current implementations
-----------------------

* Latent Dirichlet allocation
  * [Collapsed Gibbs sampling](http://nbviewer.jupyter.org/github/arongdari/python-topic-model/blob/master/notebook/LDA_example.ipynb)
  * [Variational inference](http://nbviewer.jupyter.org/github/arongdari/python-topic-model/blob/master/notebook/LDA_example.ipynb)
* Correlated topic Model
  * Variational inference
* Relational topic model (VI)
  * Exponential link function
* Author-Topic model 
* HMM-LDA
* Discrete infinite logistic normal (DILN)
  * Variational inference
* Supervised topic model
  * [Stochastic (Gibbs) EM](http://nbviewer.jupyter.org/github/arongdari/python-topic-model/blob/master/notebook/SupervisedTopicModel_example.ipynb)
  * Variational inference
* Hierarchical Dirichlet process
  * Collapsed Gibbs sampling
* Hierarchical Dirichlet scaling process

