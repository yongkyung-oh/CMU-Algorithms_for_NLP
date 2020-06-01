tagger.py: tagger for best method
tagger-bi.py: tagger for bigram model
tagger-tri.py: tagger for trigram model

train_hmm.py: train script for HMM for best method
train_tri_hmm.py: train script for HMM for bigram/trigram
train_tri_int_hmm.py: train script for HMM for bigram/trigram with interpolation smoothing
train_tri_sgt_hmm.py: train script for HMM for bigram/trigram with simple good turing smoothing

my_tag_acc.py: custom evaluation 
eval.txt: evaluation result for best method
report_bi.txt: evluation result for bigram (baseline)
report_sgt.txt: evluation result for bigram with simple good turing smoothing (best method)

ptb.23.tgs: POS tagging ptb.23.txt using best method
ptb.23-1.tgs: POS tagging ptb.23.txt using bigram with simple good turing smoothing 
ptb.23-2.tgs: POS tagging ptb.23.txt using trigram with simple good turing smoothing 