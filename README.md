This repository contains the code of the paper:  
Span-based discontinuous constituency parsing: a family of exact chart-based algorithms with time complexities from O(n^6) down to O(n^3)  
Caio Corro  
EMNLP 2020 


The cpp directory contains the parser itself.  
You must compile and install it before running the python program:  
python3 setup.py install --user


There is one script for training and one for prediction.  

The pydestruct directory contains a lot of code unrelated with this parser, it is a collection of python file I use.  
Not all code in this directory was written by myself, I tried to put link to other repos whenever I stolled it.
The file proper.prm is the file distributed with discodop.

# Reproduce experiments

You need to install to following software/libraries:

- Python 3
- Discodop (the discodop command line tool must be in the path - it is used for evaluation)
- Pytorch
- the Huggingface transformers library (I use version 3.0.1, I am not sure it will work with newer versions)

Download glove embeddings (dim=300) and put them in an embeddings directory with name glove_english.txt and glove_german.txt.
Next, put the proprocessed data in the data directory, as follows:

```
$ tree data  
data  
├── dptb  
│   ├── dev.export  
│   ├── test.export  
│   └── train.export  
├── negra  
│   ├── dev.export  
│   ├── test.export  
│   ├── train.export  
└── tiger_spmrl 
    ├── dev.export  
    ├── test.export  
    └── train.export  
```

The experiments directory contains scripts called cmd that you can execute to train and evaluate models in different settings.
If you use slurm you can execute display_results.sh that will submit all cmd scripts via sbatch.
After training is done, you can use display_results.sh to have a summary of results on test data.
I use nvidia V100 GPUs with 32GO of ram.
You will need a lot of CPU ram too (90 Go I think).


# WARNING

The inside-outside implementation **does not work**.
