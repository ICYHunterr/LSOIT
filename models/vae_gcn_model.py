# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from allennlp.modules.scalar_mix import ScalarMix
from rl_utils.RL_VAE_AR_Tree import RL_VAE_AR_Tree

from collections import defaultdict
import copy 
from rl_utils.basic import masked_softmax
from rl_utils.contrast_loss import NTXentLoss

from nltk.corpus import stopwords
import string

stopWords = set(stopwords.words('english')) | set(string.punctuation)


class DualGraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, lambda_p=0.8, bias=True):
        super(DualGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.lambda_p = lambda_p
        self.activation = nn.ReLU()
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, dep_adj, latent_adj=None, use_activation=True):
         
        hidden = torch.matmul(text, self.weight)  
        #sys.exit(0)
        denom = torch.sum(dep_adj, dim=2, keepdim=True) + 1   
        output = torch.matmul(dep_adj, hidden) / denom  
        
        
        dep_output = None 
        if self.bias is not None:
            dep_output = output + self.bias
        else:
            dep_output = output
        
        final_output = dep_output
        
        #'''
        if latent_adj is not None and self.lambda_p < 1: 
             
            denom = torch.sum(latent_adj, dim=2, keepdim=True) + 1  
            output = torch.matmul(latent_adj, hidden) / denom 
            
             
            latent_output = None 
            if self.bias is not None:
                latent_output = output + self.bias
            else:
                latent_output = output
            
            
            lambda_p = self.lambda_p# 0.5 # 0.5 for twitter  0.7 for others
            #gate =  (1-lambda_p) * latent_output.sigmoid()
            gate =  (1-lambda_p) * latent_output.sigmoid()
            
            final_output = (1.0 - gate) * dep_output + gate * latent_output
        #'''   
        if use_activation: 
            return self.activation(final_output)
        else:
            return final_output 

class GAT(nn.Module):
    """
    GAT module operated on graphs
    """
    #https://github.com/shenwzh3/RGAT-ABSA/blob/master/model_gcn.py
    def __init__(self, opt, in_dim, hidden_size=256, mem_dim=600, num_layers=2):
        super(GAT, self).__init__()
        self.opt = opt
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.dropout = nn.Dropout(opt.gcn_dropout)
        self.leakyrelu = nn.LeakyReLU(1e-2)

        self.activation = nn.ReLU(inplace=True)
        
        # Standard GAT:attention over feature
        a_layers = [
            nn.Linear(2 * mem_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)]
        
        self.afcs = nn.Sequential(*a_layers)

        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(num_layers):
            input_dim = self.in_dim if layer == 0 else mem_dim
            self.W.append(nn.Linear(input_dim, mem_dim))

    def forward(self, feature, latent_adj):
         
        B, N = latent_adj.size(0), latent_adj.size(1)
      
        # gcn layer
        for l in range(self.num_layers):
            # Standard GAT:attention over feature
            #####################################
            h = self.W[l](feature) # (B, N, D)
            #print(h.size())
            
            a_input = torch.cat([h.repeat(1, 1, N).view(
                B, N*N, -1), h.repeat(1, N, 1)], dim=2)  # (B, N*N, 2*D)
            #print(a_input.size())
            
            e = self.leakyrelu(self.afcs(a_input)).squeeze(2)  # (B, N*N)
            
            e = e.view(B, N, N)
            attention = F.softmax(e.masked_fill(latent_adj==0, -1e9), dim=-1) * latent_adj
        
            # original gat
            feature = attention.bmm(h)
            feature = self.activation(feature) #self.dropout(feature) if l < self.num_layers - 1 else feature
            #####################################
        #print("[tlog] feature: " + str(feature.size()))
        return feature


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        self.activation = nn.ReLU(inplace=True)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, dep_adj, use_activation=True):
        #print("[tlog] text: " + str(text.size()))
        hidden = torch.matmul(text, self.weight) # B * L * I,  I * O --> B * L * O 
        #print("[tlog] hidden: " + str(hidden.size()))
        #sys.exit(0)
        denom = torch.sum(dep_adj, dim=2, keepdim=True) + 1 # B * L * L 
        output = torch.matmul(dep_adj, hidden) / denom # B * L * L , B * L * O --> B * L * O
        
        dep_output = None 
        if self.bias is not None:
            dep_output = output + self.bias
        else:
            dep_output = output
        
        final_output = dep_output
        
        if use_activation: 
            return self.activation(final_output)
        else:
            return final_output 

class Classifier(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt 
        self.gc1 = DualGraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc2 = DualGraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        
        #self.gat = GAT(opt, 2*opt.hidden_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.bert_dim = 768
        #self.fc = nn.Linear(in_features=2*opt.hidden_dim,
        #                            out_features=opt.polarities_dim)
        
        self.use_output_fc = False
        if self.use_output_fc:
            self.output_fc = nn.Linear(2*opt.hidden_dim, self.bert_dim)
            self.fc = nn.Linear(self.bert_dim, opt.polarities_dim)
        else:
            self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        
        #self.distance_embeddings = nn.Embedding(100, 2*opt.hidden_dim)
        
        self.reset_parameters()

    def reset_parameters(self):
        
        torch.nn.init.uniform_(self.fc.weight, -0.002, 0.002)
        torch.nn.init.constant_(self.fc.bias, val=0)

    def mask_nonaspect(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i,0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i,1]+1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.opt.device)
        return mask*x
    
    def position_weight(self, x, aspect_double_idx, text_len, aspect_len, syntax_distance=None):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i,0]):
                if syntax_distance is None: 
                    weight[i].append(1-(aspect_double_idx[i,0]-j)/context_len)
                else:
                    weight[i].append(1-math.fabs(syntax_distance[i][j])/context_len)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i,1]+1, text_len[i]):
                if syntax_distance is None: 
                    weight[i].append(1-(j-aspect_double_idx[i,1])/context_len)
                else:
                    weight[i].append(1-math.fabs(syntax_distance[i][j])/context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight).unsqueeze(2).to(self.opt.device).float()
        return weight*x
    
    def forward(self, sentence, bert_out, adj, rl_adj, aspect_double_idx, text_len, aspect_len, syntax_distance=None, rank_logits=None):
        
       
        weighted_x = self.position_weight(sentence, aspect_double_idx, text_len, aspect_len)
        
        #'''
        x = self.gc1(weighted_x, rl_adj)
        
       
        weighted_x = x #gate_x * weighted_x  + (1.0 - gate_x) * old_weighted_x
        
        x = self.gc2(weighted_x, rl_adj) #gc2(x, rl_adj)
        
        
        gcn_x = x 
        #1
        
        aspect_x = self.mask_nonaspect(x, aspect_double_idx)
        
        alpha_mat = torch.matmul(aspect_x, sentence.transpose(1, 2))
        
        syn_dist_mask = (syntax_distance > -6).float()
        
        if bert_out is not None:
            alpha_mat2 = torch.matmul(bert_out.unsqueeze(dim=1), sentence.transpose(1, 2))
            
            alpha_mat1 = alpha_mat.sum(1, keepdim=True)
            
            alpha_mat_mixed = alpha_mat1 + alpha_mat2   # current the best 

            
            alpha_mat_mixed = alpha_mat_mixed.masked_fill(syn_dist_mask.unsqueeze(dim=1)==0, -1e9)
             
            alpha = F.softmax(alpha_mat_mixed, dim=2)
           
        else:
            alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        
        
        x = torch.matmul(alpha, sentence).squeeze(dim=1) 
        
       
        if self.use_output_fc:
            x = self.output_fc(x).tanh()

        #mlp_output = self.dropout(x)
        mlp_output = x 
        
        logits = self.fc(mlp_output)
        
        return logits, alpha.squeeze(dim=1), aspect_x.sum(dim=1), gcn_x 

class RLVAEGCN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(RLVAEGCN, self).__init__()
        print("RLVAEGCN+bert")
        self.opt = opt
       
        
        self.classifier = Classifier(opt)
        
        self.bert_dim = 768
        
        self.rl_tree_generator = RL_VAE_AR_Tree(**{'sample_num':opt.sample_num, 'hidden_dim': 2*opt.hidden_dim}) #2*opt.hidden_dim
        
        
        
        self.nt_xent_criterion = NTXentLoss(opt.device, opt.batch_size, 1.0, True)
        
        model_name = "bert-base-uncased"
        self.bert_model = BertModel.from_pretrained(model_name, output_hidden_states=True)
         
        
        
        self.text_embed_dropout = nn.Dropout(0.3) #nn.Dropout(0.3)
        self.bert_embed_dropout = nn.Dropout(0.1)
        self.use_bert_out = False

        self.bert_linear = nn.Linear(self.bert_dim, 2* opt.hidden_dim, bias=False)
         
        if self.use_bert_out:
            self.bert_fc = nn.Linear(self.bert_dim, opt.polarities_dim)
        
        nn.init.xavier_uniform_(self.bert_linear.weight)
         
        
        self.kl_div = torch.nn.KLDivLoss(reduction='none') #reduction='batchmean'
        self.count = 0
        self.mse_criterion = torch.nn.MSELoss()
        
        self.var_norm_params = {"var_normalization": True, "var": 1.0, "alpha": 0.9}
        
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        
        self.policy_trainable = True 
        
        if self.opt.use_aux_aspect:
            self.fc_aux = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)

        self.nt_xent_criterion = NTXentLoss(self.opt.device, self.opt.batch_size, 1.0, True)
        
    def debug_scalar_mix(self):
        print(self.scalar_mix.scalar_parameters)
        for param in self.scalar_mix.scalar_parameters: 
            print(param.data)
            
        print(self.scalar_mix.gamma)
        #sys.exit(0)
    def fix_policy(self):
        self.policy_trainable = False 
        for name, param in self.rl_tree_generator.named_parameters():
            print(name)
            param.requires_grad = False 
        
        self.rl_tree_generator.eval() 
        self.rl_tree_generator.training = False 
        self.rl_tree_generator.fixed = True 
        
    def get_features_for_aux_aspect(self, x, aux_aspect_targets):
        aux_batch_size = aux_aspect_targets.size(0)
        _, _, feat_size = x.size()
        aux_features = torch.zeros(aux_batch_size, feat_size, device=x.device)
         
        for i in range(aux_batch_size):
            aux_data = aux_aspect_targets[i] #(batch_index, span_start, span_end, polarity)
            batch_index = aux_data[0]
            span_start = aux_data[1]
            span_end = aux_data[2]
            aux_features[i] = torch.mean(x[batch_index, span_start: span_end+1, :], dim=0)
        
        
        return aux_features
    
    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i,0]):
                weight[i].append(1-(aspect_double_idx[i,0]-j)/context_len)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i,1]+1, text_len[i]):
                weight[i].append(1-(j-aspect_double_idx[i,1])/context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight).unsqueeze(2).to(self.opt.device).float()
        return weight*x
    
    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i,0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i,1]+1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.opt.device)
        mask_x = mask * x 
        #avg_x = (mask_x.sum(dim=1)/mask.sum(dim=1))
        sum_x = mask_x.sum(dim=1)
        return mask*x, sum_x, 1.0-mask.squeeze(dim=-1) #avg_x 

    def _normalize(self, rewards):
        if self.var_norm_params["var_normalization"]:
            with torch.no_grad():
                alpha = self.var_norm_params["alpha"]
                #print("[tlog] var: " + str(rewards.var()))
                self.var_norm_params["var"] = self.var_norm_params["var"] * alpha + rewards.var() * (1.0 - alpha)
                #print(self.var_norm_params["var"])
                #sys.exit(0)
                return rewards / self.var_norm_params["var"].sqrt().clamp(min=1.0)
        return rewards
    
    def forward(self, inputs, labels = None,  debugger=None, temperature=None):
        self.count += 1
        #self.debug_scalar_mix()
        #sys.exit(0)
        text_indices, aspect_indices, aspect_bert_indices, left_indices, left_bert_indices, adj, pos_indices, rel_indices, text_bert_indices, labeled_bert_indices, text_raw_bert_indices, bert_segments_ids, bert_token_masks, labeled_bert_segments_ids, labeled_bert_token_masks, word_seq_lengths, words, aux_aspect_targets = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        
        #print(f"[tlog] word_seq_lengths:  {word_seq_lengths}")
        #print(f"[tlog] words: {words}")
        mask = (text_indices !=0).float()
        
        _, pooled_output, encoded_layers = self.bert_model(input_ids=text_bert_indices, token_type_ids=bert_segments_ids, attention_mask=bert_token_masks)
        bert_out = None
        bert_out = self.bert_embed_dropout(pooled_output)
        bert_out = self.bert_linear(bert_out)
        #encoded_layer = self.scalar_mix(encoded_layers[1:])
        encoded_layer = encoded_layers[-1]
        batch_size, seq_len = text_indices.size()
        merged_layer = torch.zeros(batch_size, seq_len, self.bert_dim, device = text_indices.device)
        
        for b in range(batch_size):
            start_len = 1 # excluding cls
            #print(words[b], word_seq_lengths[b])
            assert len(words[b]) == len(word_seq_lengths[b])
            for i in range(len(word_seq_lengths[b])):
                merged_layer[b, i, :] = torch.mean(encoded_layer[b, start_len:start_len + word_seq_lengths[b][i], :], dim=0).squeeze(dim=0)
                start_len += word_seq_lengths[b][i]
       
        text = self.bert_linear(merged_layer)
        text_out = self.text_embed_dropout(text)
        
        if self.training: 
            _, posterior_pooled_output, posterior_encoded_layers = self.bert_model(input_ids=labeled_bert_indices, token_type_ids=labeled_bert_segments_ids, attention_mask=labeled_bert_token_masks)
            posterior_bert_out = self.bert_embed_dropout(posterior_pooled_output)
            posterior_bert_out = self.bert_linear(posterior_bert_out) #shared parameters 
            
            posterior_encoded_layer = posterior_encoded_layers[-1]
             
            posterior_merged_layer = torch.zeros(batch_size, seq_len, self.bert_dim, device = text_indices.device)
            
            for b in range(batch_size):
                start_len = 1 # excluding cls
                #print(words[b], word_seq_lengths[b])
                assert len(words[b]) == len(word_seq_lengths[b])
                for i in range(len(word_seq_lengths[b])):
                    posterior_merged_layer[b, i, :] = torch.mean(posterior_encoded_layer[b, start_len:start_len + word_seq_lengths[b][i], :], dim=0).squeeze(dim=0)
                    start_len += word_seq_lengths[b][i]
        
            posterior_text = self.bert_linear(posterior_merged_layer)
            posterior_text_out = self.text_embed_dropout(posterior_text)
        
        text_out_fixed = text_out
         
        rl_input = text_out_fixed
         
        _, aspect_vec_fixed, nonaspect_mask = self.mask(rl_input, aspect_double_idx)
        nonaspect_mask = nonaspect_mask * mask
        
        if self.training:
            _, posterior_aspect_vec_fixed, nonaspect_mask = self.mask(posterior_text_out, aspect_double_idx)
        else:
            posterior_aspect_vec_fixed = None 
            posterior_text_out = None 

        trees, samples = self.rl_tree_generator(rl_input, words, word_seq_lengths, aspect_vec_fixed, aspect_double_idx, \
                                                temperature=temperature, posterior_aspect_vecs=posterior_aspect_vec_fixed, posterior_inputs=posterior_text_out,\
                                                labels=labels)
        
        if not self.training: 
            rl_adj = torch.zeros(batch_size, seq_len, seq_len, device = text_indices.device)
            
            syn_dist =  torch.zeros(batch_size, seq_len, device = text_indices.device).fill_(-seq_len)
            
            rank_logits = torch.zeros(batch_size, seq_len, device = text_indices.device).fill_(-1e9)
            sample_rank_scores = samples['rank_scores']
            
            for b in range(batch_size):
                if debugger: 
                    print(" ".join(words[b]))
                left, right = aspect_double_idx[b][0].item(), aspect_double_idx[b][1].item()
                #print(left, right)
                if debugger: 
                    print(" ".join(words[b][left:right+1]))
                    #if self.count % 50 == 0 and b == 0:
                    #    print(" ".join(words[b]))
                    print(trees[b].print())
                pairs = []
                
                trees[b].adj(pairs, trees[b].index, 0, only_left_and_right=False) #这个地方有个bug, 都不一致
                #print(pairs)
                rank_logits[b][0:sample_rank_scores[b].size(0)] = sample_rank_scores[b]
                
                distances = {}
                trees[b].syn_distance(0, distances)
                #print(distances)
                #sys.exit(0)
                for key in distances:
                    dist = distances[key]
                    syn_dist[b][key] = dist
                
                #for key in range(left, right+1):
                #     syn_dist[b][key] = -1
                
                for pair in pairs:
                    i, j, w = pair 
                   
                    rl_adj[b][i][j] = w
                    
                        
                #print(rl_adj[b])
            logits, _, _, _  = self.classifier(text_out, bert_out, adj, rl_adj, aspect_double_idx, text_len, aspect_len, syn_dist, rank_logits)  #Batch size: 16 * 3
            #print(f"[tlog] logits: {logits.size()}")
        elif not self.policy_trainable: 
            probs, sample_trees = samples['probs'], samples['trees']
            sample_rank_scores = samples['rank_scores']
            sample_normalized_entropy = sum(samples['normalized_entropy'])
            
            rl_adj = torch.zeros(batch_size, seq_len, seq_len, device = text_indices.device)
            syn_dist =  torch.zeros(batch_size, seq_len, device = text_indices.device).fill_(-seq_len)
            
            #print(len(sample_trees))
            
            for b in range(len(sample_trees)):
                #print(" ".join(words[b]))
                left, right = aspect_double_idx[b][0].item(), aspect_double_idx[b][1].item()
                 
                pairs = []
                sample_trees[b].adj(pairs, sample_trees[b].index, 0)
                 
                distances = {}
                sample_trees[b].syn_distance(0, distances)
                
                for key in distances:
                    dist = distances[key]
                    syn_dist[b][key] = dist
                
                for pair in pairs:
                    i, j, w = pair 
                    
                    rl_adj[b][i][j] = w
            sample_logits, sample_alphas, sample_features, sample_gcn_outputs = self.classifier(text_out, bert_out, \
                                                                                                adj, rl_adj, aspect_double_idx, \
                                                                                                text_len, aspect_len, syn_dist)
            
            logits = sample_logits
            
            syn_dist = F.softmax(syn_dist*2, dim=-1)
        
            batch_attention_loss = (self.kl_div(input=(sample_alphas+1e-9).log(), target=syn_dist)).sum(dim=-1)
            attention_loss = batch_attention_loss.mean()
            
            if self.opt.use_aux_aspect and self.training and aux_aspect_targets.size(0) > 0:
                 
                aux_aspect_x = self.get_features_for_aux_aspect(sample_gcn_outputs, aux_aspect_targets) # B' * D
                #print(aux_aspect_x.size())
                #sys.exit(0)
                aux_output = self.fc_aux(aux_aspect_x)
                #print(aux_aspect_targets)
                #sys.exit(0)
                aux_loss = 0.1 * self.criterion(aux_output, aux_aspect_targets[:,-1]).mean()
                #print(aux_loss.size())
        ###########################
        else: 
            # samples prediction for REINFORCE
            #sample_logits = self.classifier(samples['h'])
            sample_num = self.opt.sample_num 
            aspect_double_idx_expanded = aspect_double_idx.unsqueeze(dim=1).repeat(1, sample_num, 1).view(batch_size * sample_num, -1)
            text_out_expanded = text_out.unsqueeze(dim=1).repeat(1, sample_num, 1, 1).view(batch_size * sample_num, seq_len, -1)
            text_len_expanded = text_len.unsqueeze(dim=1).repeat(1, sample_num).view(batch_size * sample_num)
            
            aspect_len_expanded = aspect_len.unsqueeze(dim=1).repeat(1, sample_num).view(batch_size * sample_num)
            #nonaspect_mask_expanded = nonaspect_mask.unsqueeze(dim=1).repeat(1, sample_num, 1).view(batch_size * sample_num, -1)
            mask_expanded = mask.unsqueeze(dim=1).repeat(1, sample_num, 1).view(batch_size * sample_num, -1)
            
            adj_expanded = adj.unsqueeze(dim=1).repeat(1, sample_num, 1,1).view(batch_size*sample_num, seq_len, seq_len)
            bert_out_expanded = bert_out.unsqueeze(dim=1).repeat(1, sample_num, 1).view(batch_size * sample_num, -1)
            #print(aspect_double_idx_expanded)
            #print(text_out_expanded.size())
            
            #sys.exit(0)
            # rl training loss for sampled trees
            probs, sample_trees = samples['probs'], samples['trees']
            sample_rank_scores = samples['rank_scores']
            sample_prior_scores = samples['prior_scores']
            sample_posterior_scores = samples['posterior_scores']
            sample_normalized_entropy = sum(samples['normalized_entropy'])
            
            rl_adj = torch.zeros(batch_size * sample_num, seq_len, seq_len, device = text_indices.device)
            
             
            
            syn_dist =  torch.zeros(batch_size * sample_num, seq_len, device = text_indices.device).fill_(-seq_len)
            
            rank_logits = torch.zeros(batch_size * sample_num, seq_len, device = text_indices.device).fill_(-1e9)
            
            prior_scores = torch.zeros(batch_size, seq_len, device = text_indices.device).fill_(-1e9)
            posterior_scores = torch.zeros(batch_size, seq_len, device = text_indices.device).fill_(-1e9)
            
            
            for b in range(len(sample_trees)):
                #print(" ".join(words[b]))
                left, right = aspect_double_idx_expanded[b][0].item(), aspect_double_idx_expanded[b][1].item()
                 
                debug = False  
                if debug: 
                    if self.count % 50 == 0:
                        print(" ".join(words[b//sample_num]))
                        print(" ".join(words[b//sample_num][left:right+1]))
                        print(sample_rank_scores[b])
                        print(sample_trees[b].print())
                #print(sample_trees[b].print())
                pairs = []
                sample_trees[b].adj(pairs, sample_trees[b].index, 0)
                #print(pairs)
                if b % sample_num ==0: 
                    prior_scores[b//sample_num][0:sample_prior_scores[b//sample_num].size(0)] = sample_prior_scores[b//sample_num]
                    posterior_scores[b//sample_num][0:sample_posterior_scores[b//sample_num].size(0)] = sample_posterior_scores[b//sample_num]
                    
                #rank_logits[b][0:sample_rank_scores[b].size(0)] = sample_rank_scores[b]
                rank_logits[b][0:sample_rank_scores[b].size(0)] = sample_posterior_scores[b//sample_num]
                
                distances = {}
                sample_trees[b].syn_distance(0, distances)
                #print(distances)
                #sys.exit(0)
                for key in distances:
                    dist = distances[key]
                    syn_dist[b][key] = dist
                
                 
                
                for pair in pairs:
                    i, j, w = pair 
                    
                    rl_adj[b][i][j] = w

            sample_logits, sample_alphas, sample_features, sample_gcn_outputs = self.classifier(text_out_expanded, bert_out_expanded, adj_expanded,\
                                                                                                rl_adj, aspect_double_idx_expanded, text_len_expanded,\
                                                                                                 aspect_len_expanded, syn_dist, rank_logits)
            
            reshaped_sample_logits = sample_logits.view(batch_size, sample_num, -1)
            reshaped_sample_features = sample_features.view(batch_size, sample_num, -1)
            
            
            reshaped_sample_gcn_features = sample_gcn_outputs.view(batch_size, sample_num, -1)
            #othertwo_sample_logits = reshaped_sample_logits[:,1:,:].contiguous().view(batch_size * (sample_num-1), -1)
            
            logits = reshaped_sample_logits[:,0,:]
            
             
           
            sample_label_pred = sample_logits.max(1)[1]
             
            sample_label_gt = labels.unsqueeze(1).expand(-1, sample_num).contiguous().view(-1)
             
            syn_dist = F.softmax(syn_dist*2, dim=-1)
           
            
            batch_attention_loss = (self.kl_div(input=(sample_alphas+1e-9).log(), target=syn_dist)).sum(dim=-1)
            attention_loss = batch_attention_loss.mean()
             
            batch_distill_loss =  ((self.kl_div(input=(F.softmax(rank_logits * 10.0, dim=-1)+1e-9).log(), target=sample_alphas.detach()) * mask_expanded).sum(dim=-1))
            distill_loss = batch_distill_loss.mean()
            
            
            batch_vaekl_loss =  ((self.kl_div(input=(F.softmax(prior_scores, dim=-1)+1e-9).log(), target=F.softmax(posterior_scores, dim=-1)) * mask).sum(dim=-1))
            vaekl_loss = batch_vaekl_loss.mean()
            
            
            sample_i_pairs = reshaped_sample_features[:,0,:]
            sample_j_pairs = reshaped_sample_features[:,1,:]
            
             
            
            use_ce_rewards = True  
            if use_ce_rewards: 
                ce_rewards = self.criterion(sample_logits, sample_label_gt).detach()
                
                
            
                reshaped_ce_rewards = ce_rewards.view(batch_size, sample_num)
                
                ce_mean_rewards = reshaped_ce_rewards.mean(dim=-1, keepdim=True)
                ce_normalized_rewards = (reshaped_ce_rewards - ce_mean_rewards).view(-1)
                
            
            use_prob_rewards = False 
            if use_prob_rewards: 
                rl_rewards = (F.softmax(sample_logits, dim=-1) * F.one_hot(sample_label_gt, 3)).sum(dim=-1)
                reshaped_rl_rewards = rl_rewards.view(batch_size, sample_num)
                
                rl_mean_rewards = reshaped_rl_rewards.mean(dim=-1, keepdim=True)
                rl_rewards = (reshaped_rl_rewards - rl_mean_rewards).view(-1)
            else:
                rl_rewards = torch.eq(sample_label_gt, sample_label_pred).float().detach() * 2 - 1
           
            
            if use_ce_rewards:
                 
                rl_rewards = rl_rewards + ce_normalized_rewards
             
            rl_loss = 0
            # average of word
            final_probs = defaultdict(list)
             
            
            for i in range(len(labels)):
                 
                for j in range(0, sample_num):
                    k = i * sample_num + j
                    
                    for w in probs[k]:
                        
                        items = [p*rl_rewards[k] for p in probs[k][w]]
                        
                        final_probs[w] += items 
                         
                        
            for w in final_probs:
                rl_loss += - sum(final_probs[w]) / (len(final_probs[w])) #num_counts[w] 
            
            if len(final_probs) > 0:
                rl_loss /= len(final_probs)

            rl_loss *= self.opt.rl_weight 
            
            
            if self.opt.use_aux_aspect and self.training and aux_aspect_targets.size(0) > 0:
                #print(f"[tlog] sample_gcn_outputs.size(): {sample_gcn_outputs.size()}")
                #sys.exit(0)
                reshaped_sample_gcn_outputs = sample_gcn_outputs.view(batch_size, sample_num, seq_len, -1)
                
                aux_aspect_x = self.get_features_for_aux_aspect(reshaped_sample_gcn_outputs[:,0,:,:], aux_aspect_targets) # B' * D
                #print(aux_aspect_x.size())
                #sys.exit(0)
                aux_output = self.fc_aux(aux_aspect_x)
                #print(aux_aspect_targets)
                #sys.exit(0)
                aux_loss = 0.01 * self.criterion(aux_output, aux_aspect_targets[:,-1]).mean() #adv loss 
                #print(aux_loss.size())
        ###########################
        
        if self.training:
            loss = self.criterion(sample_logits, sample_label_gt).mean()
             
            loss = loss + attention_loss * self.opt.td_weight 
            
            
            if self.policy_trainable: 
                
                vae_weight = 0.05 + self.count/100 * 0.01
                if vae_weight > 0.1: 
                    vae_weight = 0.1
                         
                loss = loss + rl_loss + self.opt.ent_weight * sample_normalized_entropy  + vae_weight * vaekl_loss + distill_loss * self.opt.att_weight #self.att_weight  
                
            if self.opt.use_aux_aspect and aux_aspect_targets.size(0) > 0:
                loss = loss + aux_loss 
            
            
            return logits, loss
        else:
            return logits
