# Unsupervised Machine Translation of English and Chinese Words
This program aligns English and Chinese words embeddings, and used unsupervised learning to train the model, and translate English words to Chinese words.

Dependencies:
1. Python 2.x
2. PyTorch. The version must be torch 0.3.1, instead of 0.4.0
3. In addition, download and install Faiss for fast nearest neighbor search for GPU. Otherwise the nearest neighbor search will be significantly slower.
conda install faiss-gpu -c pytorch

To run the program, first download the required data from amazonaws:(This is very big, so we will not include data in our respository)
1. Get English fastText Wikipedia embeddings
curl -Lo data/wiki.en.vec https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec
2. Get Chinese fastText Wikipedia embeddings
curl -Lo data/wiki.zh.vec https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.es.vec
3. Get full, train, test dictinaries for English and Chinese from aws:
curl -Lo crosslingual/dictionaries/zh-en.txt https://s3.amazonaws.com/arrival/dictionaries/zh-en.txt
curl -Lo crosslingual/dictionaries/zh-en.0-5000.txt https://s3.amazonaws.com/arrival/dictionaries/zh-en.txt
curl -Lo crosslingual/dictionaries/zh-en.5000-6500.txt https://s3.amazonaws.com/arrival/dictionaries/zh-en.txt
curl -Lo crosslingual/dictionaries/en-zh.txt https://s3.amazonaws.com/arrival/dictionaries/zh-en.txt
curl -Lo crosslingual/dictionaries/en-zh.0-5000.txt https://s3.amazonaws.com/arrival/dictionaries/zh-en.txt
curl -Lo crosslingual/dictionaries/en-zh.5000-6500.txt https://s3.amazonaws.com/arrival/dictionaries/zh-en.txt


Then open the jupyter notebook train.ipynb to run the program and look up for more details.

Note that at this time, the program is partially successful. The complete version will be release in June.