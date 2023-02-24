# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from allennlp.modules.scalar_mix import ScalarMix

from collections import defaultdict
import copy 
from rl_utils.basic import masked_softmax


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
        
        self.dropout = nn.Dropout(0.3)
        
        self.fc = nn.Linear(in_features=2*opt.hidden_dim,
                                    out_features=opt.polarities_dim)
        
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
    
    def forward(self, sentence, bert_out, adj, aspect_double_idx, text_len, aspect_len, syntax_distance=None,):
        
        '''
        if syntax_distance is not None: 
            dist_inputs = self.distance_embeddings(torch.abs(syntax_distance).long())
            #print(dist_inputs.size())
            dist_inputs = self.dropout(dist_inputs)
            sentence = sentence + dist_inputs
            #sys.exit(0)
        '''
        weighted_x = self.position_weight(sentence, aspect_double_idx, text_len, aspect_len)
        
        #'''
        x = self.gc1(weighted_x, adj)
        
       

        weighted_x = x #gate_x * weighted_x  + (1.0 - gate_x) * old_weighted_x
        
        x = self.gc2(weighted_x, adj) #gc2(x, rl_adj)
       
        gcn_x = x 
        #1,  
         
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
        
        mlp_output = x 
        
        logits = self.fc(mlp_output)
        
        return logits, alpha.squeeze(dim=1), aspect_x.sum(dim=1), gcn_x 

class DepGCNv2(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(DepGCNv2, self).__init__()
        print("DepGCNv2+bert")
        self.opt = opt
        
        self.classifier = Classifier(opt)
        
        self.bert_dim = 768
        
        model_name = "bert-base-uncased"
        # model_name = "indobenchmark/indobert-base-p2"
        self.bert_model = BertModel.from_pretrained(model_name, output_hidden_states=True, return_dict=False)
       
        self.text_embed_dropout = nn.Dropout(0.3) #nn.Dropout(0.3)
        self.bert_embed_dropout = nn.Dropout(0.1)
        self.use_bert_out = False

        self.bert_linear = nn.Linear(self.bert_dim, 2* opt.hidden_dim, bias=False)
      
        if self.use_bert_out:
            self.bert_fc = nn.Linear(self.bert_dim, opt.polarities_dim)
        
        nn.init.xavier_uniform_(self.bert_linear.weight)
       
        self.kl_div = torch.nn.KLDivLoss(reduction='none') #reduction='batchmean'
        self.count = 0
        
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        
        if self.opt.use_aux_aspect:
            self.fc_aux = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
    
        
    def get_features_for_aux_aspect(self, x, aux_aspect_targets):
        aux_batch_size = aux_aspect_targets.size(0)
        _, _, feat_size = x.size()
        aux_features = torch.zeros(aux_batch_size, feat_size, device=x.device)
        #print(f"[tlog] aux_aspect_targets: {aux_aspect_targets}")
        for i in range(aux_batch_size):
            aux_data = aux_aspect_targets[i] #(batch_index, span_start, span_end, polarity)
            batch_index = aux_data[0]
            span_start = aux_data[1]
            span_end = aux_data[2]
            aux_features[i] = torch.mean(x[batch_index, span_start: span_end+1, :], dim=0)
        
        #print(aux_aspect_targets.size())
        #print(aux_features.size())
        #sys.exit(0)
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

    
    def forward(self, inputs, labels = None,  debugger=None, temperature=None):
        self.count += 1
        #self.debug_scalar_mix()
        #sys.exit(0)
        text_indices, aspect_indices, aspect_bert_indices, left_indices, left_bert_indices, adj, pos_indices, rel_indices, text_bert_indices, text_raw_bert_indices, bert_segments_ids, bert_token_masks, word_seq_lengths, words, aux_aspect_targets, syn_dist = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        
        _, pooled_output, encoded_layers = self.bert_model(input_ids=text_bert_indices, token_type_ids=bert_segments_ids, attention_mask=bert_token_masks)
        bert_out = None
        bert_out = self.bert_embed_dropout(pooled_output)
        bert_out = self.bert_linear(bert_out)
         
        encoded_layer = encoded_layers[-1]
        batch_size, seq_len = text_indices.size()
        merged_layer = torch.zeros(batch_size, seq_len, self.bert_dim, device = text_indices.device)
        
        
        mask = (text_indices !=0).float()
        
         
        for b in range(batch_size):
            start_len = 1 # excluding cls
             
            assert len(words[b]) == len(word_seq_lengths[b])
            for i in range(len(word_seq_lengths[b])):
                merged_layer[b, i, :] = torch.mean(encoded_layer[b, start_len:start_len + word_seq_lengths[b][i], :], dim=0).squeeze(dim=0)
                start_len += word_seq_lengths[b][i]
        
        text = self.bert_linear(merged_layer)
        text_out = self.text_embed_dropout(text)
                
        syn_dist = (syn_dist.float()*(-1)).masked_fill(mask==0, -1e9) #masking 
        
        logits, sample_alphas, _, gcn_outputs  = self.classifier(text_out, bert_out, adj, aspect_double_idx, text_len, aspect_len, syn_dist)  #Batch size: 16 * 3
            
        if self.opt.use_aux_aspect and self.training and aux_aspect_targets.size(0) > 0:
                
            aux_aspect_x = self.get_features_for_aux_aspect(gcn_outputs, aux_aspect_targets) # B' * D
            
            aux_output = self.fc_aux(aux_aspect_x)
             
            aux_loss = 0.1 * self.criterion(aux_output, aux_aspect_targets[:,-1]).mean()
            
         
            
        syn_dist = F.softmax(syn_dist*2.0, dim=-1)
             
        batch_attention_loss = (self.kl_div(input=(sample_alphas+1e-9).log(), target=syn_dist)).sum(dim=-1)
        attention_loss = batch_attention_loss.mean()

        if debugger: 
             
            debugger.alpha = sample_alphas
            batch_size, _, = sample_alphas.size()
            for i in range(batch_size):
                 
                b, e = aspect_double_idx[i].cpu().numpy().tolist()
                 
                attention_list = sample_alphas[i].cpu().numpy().tolist()
                debugger.update_list(b, e, attention_list)
        
        if self.training:
            loss = self.criterion(logits, labels).mean()
             
            loss = loss + attention_loss * 0.1  
            
            if self.opt.use_aux_aspect and aux_aspect_targets.size(0) > 0:
                loss = loss + aux_loss 
            
            return logits, loss
        else:
            return logits
