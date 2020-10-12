This repository contains the code of the paper:  
Span-based discontinuous constituency parsing: a family of exact chart-based algorithms with time complexities from O(n^6) down to O(n^3)  
Caio Corro  
EMNLP 2020 


The cpp directory contains the parser itself.  
You must compile and install it before running the python program:  
python3 setup.py install --user


There is one script for training and one for prediction.  
Unfortunately, due to covid conditions I am really busy with my courses these days so I don't have time to check the exact cmd line to train the same model as in the paper (see the appendix for hyperparameters).  
In config.py you will find arguments used to train several variants (this is a configurations used in another software, but it contains most stuff useful).  
Don't forget to use -max-word-len 20 to speed training/prediction time.


Training loss:  
There are several training loss implemented and I can't remember which was the one use in the experiments in the paper (definitely not the margin loss).  
In train_disc_biaffine.py the line 199 is commented, I don't know why.  
I think the correct way to train the parser is only with --mean-loss and **without** --nll-loss or --margin-loss.


I apologize for this mess, I will correct this as soon as I have more free time.  
I hope this code can help people trying to reproduce experiments in the paper.


The pydestruct directory contains a lot of code unrelated with this parser, it is a collection of python file I use.  
Not all code in this directory was written by myself, I tried to put link to other repos whenever I stolled it.



# Example

train_disc_biaffine.py
--model model
--dev dev.export
--train train.export
--max-word-len 20
--epochs 200
--batch-clusters 32
--batch 5000
--storage-device cpu
--device gpu
--char-lstm-boundaries
--mean-loss
--min-word-freq 2
--char-embs
--char-embs-dim 64
--char-lstm-dim 100
--word-embs
--word-embs-dim 300
--dropout-features 0.3
--dropout-char-lstm-input 0.3
--mlp-dropout 0.33
--biaffine
--proj-dim 500
--label-proj-dim 100
--span-proj-dim 500
--tagger
--tagger-stack 1
--lstm-dim 800
--lstm-layers 1
--lstm-stacks 2
--lstm-dropout 0.3
--optim adam
--optim-lr 1e-3
--optim-lr-scheduler exponential
--optim-lr-scheduler-step 5000
--optim-lr-scheduler-decay 0.75


# other options

To use pretrain word embeddings:  
--pretrained-word-embs
--pretrained-word-embs-finetune
--pretrained-word-embs-path PARH
--pretrained-word-embs-dim 300


To use Bert (you should probably remove --char-embs option if you use Bert):  
--bert
--bert-tokenizer BERT_TOKENIZER
--bert-model BERT_MODEL
--bert-start-features
--bert-cache BERT_CACHE_PATH
--bert-mix-layers 4-7


If its a lower case Bert model:  
--bert-do-lower-case


If its an english Bert:  
--bert-split-nt


# WARNING

The inside-outside implementation **does not work**.
