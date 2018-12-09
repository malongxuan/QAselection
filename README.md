# QAselection
# For Question answering sentence selection(wikiQA,TREC-QA,InsuranceQA,SelQA,etc.)

# theano 0.82
# python 2.7
# keras 1.0

# run this code with:
# THEANO_FLAGS="device=gpu0,floatX=float32,dnn.conv.algo_bwd_filter=deterministic,dnn.conv.algo_bwd_data=deterministic" python main.py -t trecqa -m listwise -d 300 -e 10 -l 0.001 -b 6

# the code is based on "Dynamic clipped attention"
