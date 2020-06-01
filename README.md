# CMU-Algorithms_for_NLP
Repository for Algorithms for NLP (Spring 2020) - Carnegie Mellon University

Course Website(http://demo.clab.cs.cmu.edu/algo4nlp20/)

## [HW1](https://github.com/yongkyung-oh/CMU-Algorithms_for_NLP/tree/master/hw1_handout)
POS tagging forms an important part of NLP workflow for most modern day NLP systems. Within the NLP
community, POS tagging is largely perceived as a solved problem, or at least well enough solved such that most people don’t put much effort into improving POS tagging for its own sake.
I use the hidden markov model (HMM) and viterbi algorithm to conduct POS tagging. I tried two directions. First, I develop own tri-gram model using HMM. Second I tried to improve the bi-gram / tri-gram viterbi model. To deal with the unsean data, I use the linear interpolation and good-turing estimator. Overall performance is slightly improved and I can see the way of how to improve the POS tagging.

## [HW2](https://github.com/yongkyung-oh/CMU-Algorithms_for_NLP/tree/master/hw2_handout)
Sentiment classification, detecting if a piece of text is positive or negative, is a common NLP task that is useful for understanding feedback in product reviews, user's opinions, etc. Sentiment can be expressed in natural language in both trivial and non-trivial ways.
In this assignment, I build two sentiment classifiers based on Naive Bayes and neural networks. Because of the lack of train data, I conducted pre-process and random sampling. After that, I develop two classifiers. For Naive Bayes, I use the TF-IDF as feature of train data. The main model is based on MultinomialNB. In case of neural net, I compared word embedding from train data and pretrained embedding(Glove 6B). The main model is based on bi-directional LSTM. Both classifiers work well and acheive the target accuracy. 
