import torch
from torch import nn
from torch.nn import init
from torch.nn.functional import log_softmax, softmax
from collections import defaultdict

from . import basic
from .basic import TriPadLSTMLayer, Node
import numpy as np
import random
import math 


from torch.distributions import utils as distr_utils


class RL_AR_Tree(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        
        self.sample_num = kwargs.get('sample_num', 3) 
        hidden_dim = self.hidden_dim = kwargs['hidden_dim']

        
        rank_dim = 128*2        
        self.hidden2rank = nn.Linear(in_features=hidden_dim, out_features=rank_dim, bias=False)
        
        self.use_dot_product = False   
        if self.use_dot_product:     
            self.rank = nn.Sequential(
                    nn.ReLU(),
                    #nn.Tanh(),
                    #self.dropout,
                    nn.Linear(in_features=rank_dim, out_features=hidden_dim, bias=False),
                )
        else:
            self.apsect2rank = nn.Linear(in_features=hidden_dim, out_features=rank_dim, bias=False)
            self.rank = nn.Sequential(
                nn.ReLU(),
                #nn.Tanh(),
                #self.dropout,
                nn.Linear(in_features=rank_dim, out_features=1, bias=False),
            )

        self.reset_parameters()
        self.kl_div = torch.nn.KLDivLoss(reduction='none')
        self.fixed = False 
        
    def reset_parameters(self):
        
        #self.treelstm_layer.reset_parameters()
        for layer in self.rank:
            if type(layer)==nn.Linear:
                init.kaiming_normal_(layer.weight.data)
                
        init.kaiming_normal_(self.hidden2rank.weight.data)
        
        if not self.use_dot_product: 
            init.kaiming_normal_(self.apsect2rank.weight.data)
        
    def calc_score(self, x, aspect_vec=None, parent_vec=None):
        
        if self.use_dot_product: 
            assert aspect_vec is not None 
            hx = self.rank(self.hidden2rank(x))
            #print(hx.size()) #[len, hidden_size]
            #print(aspect_vec.size()) #[1, hidden_size]
            #sys.exit(0)
            s = torch.matmul(hx, aspect_vec.transpose(1, 0)).sum(dim=-1)
        else:
            hx = self.hidden2rank(x)
            if aspect_vec is not None:
                ax = self.apsect2rank(aspect_vec)
                hx = hx + ax 
            if parent_vec is not None: 
                px = self.parent2rank(parent_vec)
                hx = hx + px.unsqueeze(dim=0) 
            s = self.rank(hx)
        #'''
        #s = torch.matmul(x, aspect_vec.transpose(1, 0))
        s = s.squeeze(dim=-1) # [n, 1] -> n
        #s = softmax(s) * s 
        return s

    def greedy_build(self, sentence, embedding_scores, start, end, collector, aspect_vec=None, parent_vec=None):
        """
        Args:
            hs: (length, 1, hidden_dim)
            cs: (length, 1, hidden_dim)
            start: int
            end: int
            collector: dict
        Output:
            h, c: (1, hidden_dim), embedding of sentence[start:end]
            all probabilities 
        """
        
        if end == start:
            return None
        elif end == start+1:
            root = Node(sentence[start], [start])
            return root 
        
         
        scores = embedding_scores[start:end]  

         
        pos = start + torch.max(scores, dim=0)[1].item()
        word = sentence[pos]
        
        #if self.training and random.random() < 0.5: 
        collector['probs'][word].append((end - start) * log_softmax(scores, dim=0)[pos-start])

        probs = softmax(scores, dim=-1)
        normalized_entropy = (probs * torch.log(probs+1e-9)).sum(dim=-1) / (end-start+1)
         
        
        collector['normalized_entropy'].append(normalized_entropy)
         
        left_tree = self.greedy_build(sentence, embedding_scores, start, pos, collector)#, aspect_vec, embedding[pos])
        right_tree = self.greedy_build(sentence, embedding_scores,  pos+1, end, collector)#, aspect_vec, embedding[pos])
         
        root = Node(word, [pos], left_tree, right_tree)
         
        return root

    def greedy_build_with_aspect(self, sentence, embedding_scores, start, end, collector, aspect_vec, left, right, parent_vec=None):
         
        
        left_tree = self.greedy_build(sentence, embedding_scores, start, left, collector)#, aspect_vec, parent_vec)
        right_tree = self.greedy_build(sentence, embedding_scores, right+1, end, collector)#, aspect_vec, parent_vec)
        
        tree = Node(" ".join(sentence[left:right+1]), [i for i in range(left, right+1)], left_tree, right_tree)
        return tree 
    
    def sample_with_aspect(self, sentence, embedding_scores, start, end, collector, aspect_vec, left, right, temperature=0.2, parent_vec=None, use_binary_tree=False):
         
        left_tree = self.sample(sentence, embedding_scores,  start, left, collector, temperature=temperature, depth=1, use_binary_tree=use_binary_tree)#, aspect_vec, parent_vec)
        right_tree = self.sample(sentence, embedding_scores, right+1, end, collector, temperature=temperature, depth=1, use_binary_tree=use_binary_tree)#, aspect_vec, parent_vec)
         
        tree = Node(" ".join(sentence[left:right+1]), [i for i in range(left, right+1)], left_tree, right_tree)
        return tree 
        

    def sample(self, sentence, embedding_scores, start, end, collector, temperature=0.2, depth=0, aspect_vec=None, parent_vec=None, use_binary_tree=False):
        """
        To sample a tree structure for REINFORCE.
        """
        if end == start:
            return None
        elif end == start+1:
            root = Node(sentence[start], [start])
            return root 
        
        
        
        scores = embedding_scores[start:end]
        original_scores = scores 
        
        with torch.no_grad():
            uniforms = torch.empty_like(scores).uniform_() #1.1920928955078125e-07
             
            uniforms = distr_utils.clamp_probs(uniforms)
            gumbel_noise = -(-uniforms.log()).log()
            scores = scores + gumbel_noise
        
        
        current_temperature = temperature
        probs = softmax(scores /current_temperature, dim=0) #5.0
        
        
        if not use_binary_tree:     
            epsilon = 0.1 
            epsilon_prob = random.random()  
            if epsilon_prob < epsilon: 
                pos = random.randint(start, end-1)
            else: 
                cum = 0
                p = random.random()
                pos = end - 1
                for i in range(start, end):
                    cum = cum + probs[i-start].item()
                    if cum >= p:
                        pos = i
                        break
        else:
            pos = start + (end-start)//2
            
        word = sentence[pos]
        
        #if self.training and random.random() < 0.5: 
        collector['probs'][word].append((end - start) * torch.log(1e-9 + probs[pos-start]))  # collect log-probability of pos-th word
        
        
        #normalized_entropy = (probs * torch.log(probs)).sum(dim=-1) / (end-start+1)
        normalized_entropy = (softmax(original_scores, dim=0) * torch.log_softmax(original_scores, dim=0)).sum(dim=-1) / (end-start+1)
        
       
        
        collector['normalized_entropy'].append(normalized_entropy)
        #print(collector)
        left_tree = self.sample(sentence, embedding_scores, start, pos, collector, temperature=temperature, depth=depth+1, use_binary_tree=use_binary_tree)#, aspect_vec, embedding[pos])
        right_tree = self.sample(sentence, embedding_scores, pos+1, end, collector, temperature=temperature, depth=depth+1, use_binary_tree=use_binary_tree)#, aspect_vec, embedding[pos])
        root = Node(word, [pos], left_tree, right_tree)
        return root


    def forward(self, sentence_embedding, sentence_word, length, aspect_vecs, aspect_double_idx=None, prior_scores=None, temperature=0.2, initial_rank_scores=None):
        """
        Args:
            sentence_embedding: (batch_size, max_length, word_dim). word embedding
            sentence_word: (batch_size, max_length). word id
            length: (batch_size, ). sentence length
        """
        batch_size, max_length, _ = sentence_embedding.size()
         
        if temperature is None: 
            temperature = 0.2 
         
        structure, samples = [], {}
        samples['probs'], samples['trees'] = [], []
        samples['rank_scores'] = []
        samples['normalized_entropy'] = []
        
        
        # iterate each sentence
        for i in range(batch_size):
            sentence = sentence_word[i]
             
            embedding = sentence_embedding[i][:len(sentence)]
            
            collector = {}
            collector['probs'] = defaultdict(list)
            collector['normalized_entropy'] = []
            #print(collector)
            
            
            aspect_vec, parent_vec  = None, None 
            if aspect_double_idx is not None: 
                left, right = aspect_double_idx[i][0].item(), aspect_double_idx[i][1].item()
                
                aspect_vec = aspect_vecs[i].unsqueeze(dim=0)
                
            if prior_scores is None: 
                if aspect_vec is not None: 
                    embedding_scores = self.calc_score(embedding, aspect_vec).squeeze(dim=-1)
                else:
                    embedding_scores = self.calc_score(embedding).squeeze(dim=-1)
            else:
                embedding_scores = prior_scores[i][:len(sentence)]
            
            
            if aspect_double_idx is None: 
                tree = self.greedy_build(sentence, embedding_scores, 0, len(sentence), collector)#, aspect_vec, parent_vec)
            else:
                tree = self.greedy_build_with_aspect(sentence, embedding_scores, 0, len(sentence), collector, aspect_vec, left, right)#, parent_vec)
                
                
           
            
            structure.append(tree)
            
            ##################################
            # Monte Carlo
             
            for j in range(self.sample_num):
                if j > 0: # if j==0, just use the state+probs from greedy_build
                    #probs = defaultdict(list)
                    collector = {}
                    collector['probs'] = defaultdict(list)
                    collector['normalized_entropy'] = []
                    #tree = self.sample(sentence, embedding_scores, 0, len(sentence), collector)
                    if aspect_double_idx is None: 
                        tree = self.sample(sentence, embedding_scores, 0, len(sentence), collector)#, aspect_vec, parent_vec)
                    else:
                        tree = self.sample_with_aspect(sentence, embedding_scores, 0, len(sentence), collector, aspect_vec, left, right, temperature=temperature)#, use_binary_tree=(j==1))#, parent_vec)
                #print("===========")
                #print(collector)
                samples['probs'].append(collector['probs']) # a list of dict of Variable
                samples['trees'].append(tree)
                samples['rank_scores'].append(embedding_scores)
                if j == 0: 
                    samples['normalized_entropy'].append(sum(collector['normalized_entropy']))
                if not self.training or prior_scores is not None or self.fixed:
                    #sys.exit(0)
                    break
         
        return structure, samples

    
    def _get_kl_loss(self, tree, rank_logits, target_logits, start, end, collector):
        
        if start == end: 
            return 
        
        pos = tree.index[0]
        
        if tree.left is not None: 
            self._get_kl_loss(tree.left, rank_logits[:pos-start+1], target_logits[:pos-start+1], start, pos, collector)
            
        if tree.right is not None: 
            self._get_kl_loss(tree.right, rank_logits[pos+1:], target_logits[pos+1:], pos+1, end, collector)
        
        local_kl_loss = self.kl_div(input=(softmax(rank_logits * 10.0, dim=-1)+1e-9).log(), target=softmax(target_logits, dim=-1)).sum(dim=-1)
        
        collector['kl_loss'].append(local_kl_loss.unsqueeze(dim=0))
        collector['kl_weight'].append((end-start))
    
    def get_kl_loss(self, tree, rank_logits, target_probs, text_len, aspect_double_idx):
        
        rank_logits = rank_logits[:text_len.item()]
        target_probs = target_probs.detach()[:text_len.item()]
        target_logits = torch.log(target_probs[:text_len.item()] + 1e-9)
        
        collector = {}
        collector['kl_loss'] = list()
        collector['kl_weight'] = list()
        
        left, right = aspect_double_idx[0].item(), aspect_double_idx[1].item()
        
        self._get_kl_loss(tree.left, rank_logits[0:left+1], target_logits[0:left+1], 0, left, collector)
        self._get_kl_loss(tree.right, rank_logits[right+1:text_len], target_logits[right+1: text_len], right+1, text_len, collector)
        
        local_kl_loss = (self.kl_div(input=(softmax(rank_logits * 10.0, dim=-1)+1e-9).log(), target=target_probs)).sum(dim=-1)
        
        collector['kl_loss'].append((text_len-1) * local_kl_loss.unsqueeze(dim=0))
        collector['kl_weight'].append(text_len-1)
        #print(collector)
        
        kl_loss = torch.cat(collector['kl_loss'], dim=0).sum()/ sum(collector['kl_weight'])
        
        return kl_loss
        
