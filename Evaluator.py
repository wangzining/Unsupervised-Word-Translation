import numpy as np
import os
import io
import torch

from Faiss_NN import get_nn_avg_dist

class Evaluator(object):

    def __init__(self, trainer):
        """
        Initialize evaluator.
        """
        self.src_emb = trainer.src_emb
        self.tgt_emb = trainer.tgt_emb
        self.src_dic = trainer.src_dic
        self.tgt_dic = trainer.tgt_dic
        self.mapping = trainer.mapping
        self.discriminator = trainer.discriminator
        self.param_list = trainer.param_list
        
        
        
    def get_word_translation_accuracy(self, word2id1, src_emb, word2id2, tgt_emb, method):
        
        # Load
        path = "crosslingual/dictionaries/en-zh.5000-6500.txt"
        pairs = []
        not_found = 0
        not_found1 = 0
        not_found2 = 0

        with io.open(path, 'r', encoding='utf-8') as f:
            for _, line in enumerate(f):
                assert line == line.lower()
                word1, word2 = line.rstrip().split()
                if word1 in word2id1 and word2 in word2id2:
                    pairs.append((word1, word2))
                else:
                    not_found += 1
                    not_found1 += int(word1 not in word2id1)
                    not_found2 += int(word2 not in word2id2)
        print("Loaded %i pairs of words in the dictionary (%i unique). "
                "%i other pairs contained at least one unknown word "
                "(%i in lang1, %i in lang2)"
                % (len(pairs), len(set([x for x, _ in pairs])),
                   not_found, not_found1, not_found2))
        #sort and return
        pairs = sorted(pairs, key=lambda x: word2id1[x[0]])
        dico = torch.LongTensor(len(pairs), 2)
        for i, (word1, word2) in enumerate(pairs):
            dico[i, 0] = word2id1[word1]
            dico[i, 1] = word2id2[word2]

        
        # to cuda
        if self.param_list.useGPU:
            dico = dico.cuda()
        if dico[:, 0].max() >= src_emb.size(0):
            print("dico[:, 0].max() (",dico[:, 0].max(),") should be < src_emb.size(0) (",src_emb.size(0),")")
            
        if dico[:, 1].max() >= tgt_emb.size(0):
            print("dico[:, 1].max() (",dico[:, 1].max(),") should be < tgt_emb.size(0) (",tgt_emb.size(0),")")
            
        # normalize word embeddings
        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
        
        if method == "nn":
            print("Using nn for matching pairs")
            query = src_emb[dico[:, 0]]
            scores = query.mm(tgt_emb.transpose(0, 1))  #Performs a matrix multiplication
        else: #CSLS10
            knn = self.param_list.csls_k
            print("Using CSLS with k = ", knn)
            
            average_dist1 = get_nn_avg_dist(tgt_emb, src_emb, knn)
            average_dist2 = get_nn_avg_dist(src_emb, tgt_emb, knn)
            average_dist1 = torch.from_numpy(average_dist1).type_as(src_emb)
            average_dist2 = torch.from_numpy(average_dist2).type_as(tgt_emb)
            # queries / scores
            query = src_emb[dico[:, 0]]
            scores = tgt_emb.mm(emb2.transpose(0, 1))
            scores.mul_(2)
            scores.sub_(average_dist1[dico[:, 0]][:, None] + average_dist2[None, :])
            
    
    
        
    def word_translation(self):
        src_emb = self.mapping(self.src_emb.weight).data
        tgt_emb = self.tgt_emb.weight.data

        for method in ['nn', 'csls_knn_10']:
            results = self.get_word_translation_accuracy(self.src_dic.word2id, src_emb,
                self.tgt_dic.word2id, tgt_emb, method)
        #print([('%s-%s' % (k, method), v) for k, v in results])
        


        
    
    def evaluate(self):
        #self.monolingual_wordsim(to_log)
        #self.crosslingual_wordsim(to_log)
        self.word_translation()
        #self.sent_translation(to_log)
        #self.dist_mean_cosine(to_log)
        
        # Evaulate translation accuracy of our mapping
        
        
        
        


        