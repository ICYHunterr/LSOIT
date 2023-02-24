# -*- coding: utf-8 -*-
import sys
sys.path.append(".")
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from allennlp.modules.scalar_mix import ScalarMix
from rl_utils.RL_AR_Tree import RL_AR_Tree
from collections import defaultdict
import copy 
from rl_utils.basic import masked_softmax
from rl_utils.contrast_loss import NTXentLoss
from torch.autograd import Variable
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

    def forward(self, input, latent_adj=None, use_activation=True):
        text, dep_adj = input
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
            return (self.activation(final_output), dep_adj)
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
        if opt.num_layers == 2:
            self.gc1 = DualGraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
            self.gc2 = DualGraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        else:
            gc = []
            for i in range(opt.num_layers):
                gc.append(DualGraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim))
            self.gc = nn.Sequential(*gc)

        #self.gat = GAT(opt, 2*opt.hidden_dim)
        
        self.dropout = nn.Dropout(0.3)
        
        self.fc = nn.Linear(in_features=2*opt.hidden_dim,
                                    out_features=opt.polarities_dim)
        
        
        
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
        if self.opt.num_layers == 2:
            x = self.gc1((weighted_x, rl_adj))[0]
            weighted_x = x #gate_x * weighted_x  + (1.0 - gate_x) * old_weighted_x
            x = self.gc2((weighted_x, rl_adj))[0] #gc2(x, rl_adj)
        else:
            x = self.gc((weighted_x, rl_adj))[0]
         
    
        
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

class RLGCN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(RLGCN, self).__init__()
        print("RLGCN+bert")
        self.opt = opt
        
        self.classifier = Classifier(opt)
        
        self.bert_dim = 768
        
        self.rl_tree_generator = RL_AR_Tree(**{'sample_num':opt.sample_num, 'hidden_dim': 2*opt.hidden_dim}) #2*opt.hidden_dim
        
        self.nt_xent_criterion = NTXentLoss(opt.device, opt.batch_size, 1.0, True)
        
        model_name = "bert-base-uncased"
        # model_name = "indobenchmark/indobert-base-p2"
        self.bert_model = BertModel.from_pretrained(model_name, output_hidden_states=True, return_dict=False)
        if opt.NoisyTune:
            for name, para in self.bert_model.named_parameters():
                self.bert_model.state_dict()[name][:] += (torch.rand(para.size())-0.5) * opt.noise_lambda * torch.std(para)
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
            
        self.dist_file = open("rest16.rlgcn.dist", "w")
        # self.add_activation = nn.ReLU()
        self.add_activation = nn.Sigmoid()
        self.attention_activation = nn.ReLU()
        self.gate_activation = nn.Sigmoid()
        self.sememe_activation = nn.ReLU()
        if opt.use_const == "add":
            self.const_embed = nn.Embedding(opt.const_size, opt.const_dim, padding_idx=0)
            self.const_pos_embed = nn.Embedding(opt.const_pos_size, opt.const_dim, padding_idx=0)
            self.const_W = nn.Linear(opt.const_dim, 4* opt.hidden_dim)
            self.context_W = nn.Linear(self.bert_dim, 4* opt.hidden_dim)
            self.const_fc = nn.Linear(4*opt.hidden_dim, 2* opt.hidden_dim)

        if opt.use_const == "concat_linear":
            self.const_embed = nn.Embedding(opt.const_size, opt.const_dim, padding_idx=0)
            self.const_pos_embed = nn.Embedding(opt.const_pos_size, opt.const_dim, padding_idx=0)
            self.concat_const = nn.Linear(self.bert_dim+opt.const_dim, self.bert_dim+opt.const_dim)
            self.concat_const_fc = nn.Linear(self.bert_dim+opt.const_dim, 2* opt.hidden_dim)
        
        if opt.use_const == "attention":
            self.const_embed = nn.Embedding(opt.const_size, opt.const_dim, padding_idx=0)
            self.const_pos_embed = nn.Embedding(opt.const_pos_size, opt.const_dim, padding_idx=0)
            self.const_K = nn.Linear(opt.const_dim, 2* opt.hidden_dim)
            self.const_Q = nn.Linear(self.bert_dim, 2* opt.hidden_dim)
            self.const_V = nn.Linear(self.bert_dim, 2* opt.hidden_dim)
            # self.layer_norm = nn.LayerNorm(2* opt.hidden_dim)

        if opt.use_const == "concat_attention":
            self.const_embed = nn.Embedding(opt.const_size, opt.const_dim, padding_idx=0)
            self.const_pos_embed = nn.Embedding(opt.const_pos_size, opt.const_dim, padding_idx=0)
            self.const_K = nn.Linear(opt.const_dim, 2* opt.hidden_dim)
            self.const_Q = nn.Linear(self.bert_dim + opt.const_dim, 2* opt.hidden_dim)
            self.const_V = nn.Linear(self.bert_dim + opt.const_dim, 2* opt.hidden_dim)
            # self.layer_norm = nn.LayerNorm(2* opt.hidden_dim)
        
        if opt.use_const == "concat_gate_attention":
            self.const_embed = nn.Embedding(opt.const_size, opt.const_dim, padding_idx=0)
            self.const_pos_embed = nn.Embedding(opt.const_pos_size, opt.const_dim, padding_idx=0)
            self.const_G = nn.Linear(opt.const_dim, 2* opt.hidden_dim)
            self.const_K = nn.Linear(opt.const_dim, 2* opt.hidden_dim)
            self.const_Q = nn.Linear(2* opt.hidden_dim, 2* opt.hidden_dim)
            self.const_V = nn.Linear(self.bert_dim + opt.const_dim, 2* opt.hidden_dim)
            # self.layer_norm = nn.LayerNorm(2* opt.hidden_dim)
        
        if opt.use_const == "gate":
            self.const_embed = nn.Embedding(opt.const_size, opt.const_dim, padding_idx=0)
            self.const_pos_embed = nn.Embedding(opt.const_pos_size, opt.const_dim, padding_idx=0)
            self.const_G = nn.Linear(opt.const_dim, 2* opt.hidden_dim)
            # self.layer_norm = nn.LayerNorm(2* opt.hidden_dim)

        if opt.use_const == "pyramid2_layer_norm_drop":

            b_c_dim = self.bert_dim + opt.const_dim
            self.bc = nn.Linear(b_c_dim, opt.pyramid_hidden_dim)

            sum_dim = concat_dim = opt.pyramid_hidden_dim

            self.p1 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim
            self.p2 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim
            self.p3 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim

            self.w0 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w1 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w2 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w3 = nn.Parameter(torch.tensor(1.0)).cuda()
            

            self.const_embed = nn.Embedding(opt.const_size, opt.const_dim, padding_idx=0)
            self.const_pos_embed = nn.Embedding(opt.const_pos_size, opt.const_dim, padding_idx=0)


            self.cs_drop = nn.Dropout(opt.cs_dropout)
            self.linear_drop = nn.Dropout(opt.linear_dropout)

            self.concat_cs = nn.Linear(concat_dim, 2* opt.hidden_dim)
            self.layer_norm = nn.LayerNorm(2* opt.hidden_dim)


        if opt.use_dep == "pyramid2_layer_norm_drop":
            b_d_dim = self.bert_dim + opt.dep_dim
            self.bd = nn.Linear(b_d_dim, opt.pyramid_hidden_dim)

            sum_dim = concat_dim = opt.pyramid_hidden_dim

            self.p1 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim
            self.p2 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim
            self.p3 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim

            self.w0 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w1 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w2 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w3 = nn.Parameter(torch.tensor(1.0)).cuda()
            

            self.dep_embed = nn.Embedding(opt.dep_size, opt.dep_dim, padding_idx=0)
            self.dep_pos_embed = nn.Embedding(opt.dep_pos_size, opt.dep_dim, padding_idx=0)


            self.cs_drop = nn.Dropout(opt.cs_dropout)
            self.linear_drop = nn.Dropout(opt.linear_dropout)

            self.concat_cs = nn.Linear(concat_dim, 2* opt.hidden_dim)
            self.layer_norm = nn.LayerNorm(2* opt.hidden_dim)

        if opt.use_sememe == "concat_linear":
            self.sememe_embed = nn.Embedding(opt.sememe_size, opt.sememe_dim, padding_idx=0)
            self.concat_sememe = nn.Linear(self.bert_dim + opt.sememe_dim, self.bert_dim + opt.sememe_dim)
            self.concat_sememe_fc = nn.Linear(self.bert_dim + opt.sememe_dim, 2* opt.hidden_dim)

        
        if opt.use_sememe == "attention":
            self.sememe_embed = nn.Embedding(opt.sememe_size, opt.sememe_dim, padding_idx=0)
            self.sememe_K = nn.Linear(opt.sememe_dim, 2*opt.hidden_dim)
            self.sememe_Q = nn.Linear(self.bert_dim, 2*opt.hidden_dim)
            self.sememe_V = nn.Linear(self.bert_dim, 2*opt.hidden_dim)

        if opt.use_sememe == "gate":
            self.sememe_embed = nn.Embedding(opt.sememe_size, opt.sememe_dim, padding_idx=0)
            self.sememe_G = nn.Linear(opt.sememe_dim, 2* opt.hidden_dim)
            # self.layer_norm = nn.LayerNorm(2* opt.hidden_dim)

        if opt.use_const_and_sememe == "attention":
            self.const_embed = nn.Embedding(opt.const_size, opt.const_dim, padding_idx=0)
            self.const_pos_embed = nn.Embedding(opt.const_pos_size, opt.const_dim, padding_idx=0)
            self.const_K = nn.Linear(opt.const_dim, 2* opt.hidden_dim)
            self.const_Q = nn.Linear(self.bert_dim, 2* opt.hidden_dim)
            self.const_V = nn.Linear(self.bert_dim, 2* opt.hidden_dim)
            # self.layer_norm = nn.LayerNorm(2* opt.hidden_dim)

            self.sememe_embed = nn.Embedding(opt.sememe_size, opt.sememe_dim, padding_idx=0)
            self.sememe_K = nn.Linear(opt.sememe_dim, 2*opt.hidden_dim)
            self.sememe_Q = nn.Linear(self.bert_dim, 2*opt.hidden_dim)
            self.sememe_V = nn.Linear(self.bert_dim, 2*opt.hidden_dim)
        if opt.use_const_and_sememe == "attention_FFN":
            self.const_embed = nn.Embedding(opt.const_size, opt.const_dim, padding_idx=0)
            self.const_pos_embed = nn.Embedding(opt.const_pos_size, opt.const_dim, padding_idx=0)
            self.const_K = nn.Linear(opt.const_dim, 2* opt.hidden_dim)
            self.const_Q = nn.Linear(self.bert_dim, 2* opt.hidden_dim)
            self.const_V = nn.Linear(self.bert_dim, 2* opt.hidden_dim)
            

            self.sememe_embed = nn.Embedding(opt.sememe_size, opt.sememe_dim, padding_idx=0)
            self.sememe_K = nn.Linear(opt.sememe_dim, 2*opt.hidden_dim)
            self.sememe_Q = nn.Linear(self.bert_dim, 2*opt.hidden_dim)
            self.sememe_V = nn.Linear(self.bert_dim, 2*opt.hidden_dim)

            
            self.sc_K = nn.Linear(opt.sememe_dim + opt.const_dim, 2*opt.hidden_dim)
            self.sc_Q = nn.Linear(self.bert_dim, 2*opt.hidden_dim)
            self.sc_V = nn.Linear(self.bert_dim, 2*opt.hidden_dim)

            self.fc1 = nn.Linear(3*2*opt.hidden_dim, 3*2*opt.hidden_dim * 2)
            self.fc2 = nn.Linear(3*2*opt.hidden_dim * 2, 2*opt.hidden_dim)
            self.drop = nn.Dropout(0.1)

            self.layer_norm = nn.LayerNorm(2* opt.hidden_dim)
        if opt.use_const_and_sememe == "attention_add":
            self.const_embed = nn.Embedding(opt.const_size, opt.const_dim, padding_idx=0)
            self.const_pos_embed = nn.Embedding(opt.const_pos_size, opt.const_dim, padding_idx=0)
            self.const_K = nn.Linear(opt.const_dim, 2* opt.hidden_dim)
            self.const_Q = nn.Linear(self.bert_dim, 2* opt.hidden_dim)
            self.const_V = nn.Linear(self.bert_dim, 2* opt.hidden_dim)
            

            self.sememe_embed = nn.Embedding(opt.sememe_size, opt.sememe_dim, padding_idx=0)
            self.sememe_K = nn.Linear(opt.sememe_dim, 2*opt.hidden_dim)
            self.sememe_Q = nn.Linear(self.bert_dim, 2*opt.hidden_dim)
            self.sememe_V = nn.Linear(self.bert_dim, 2*opt.hidden_dim)

            
            self.sc_K = nn.Linear(opt.sememe_dim + opt.const_dim, 2*opt.hidden_dim)
            self.sc_Q = nn.Linear(self.bert_dim, 2*opt.hidden_dim)
            self.sc_V = nn.Linear(self.bert_dim, 2*opt.hidden_dim)

            self.drop = nn.Dropout(0.1)

            self.layer_norm = nn.LayerNorm(2* opt.hidden_dim)
        
        if opt.use_const_and_sememe == "attention_drop":
            self.const_embed = nn.Embedding(opt.const_size, opt.const_dim, padding_idx=0)
            self.const_pos_embed = nn.Embedding(opt.const_pos_size, opt.const_dim, padding_idx=0)
            self.const_K = nn.Linear(opt.const_dim, 2* opt.hidden_dim)
            self.const_Q = nn.Linear(self.bert_dim, 2* opt.hidden_dim)
            self.const_V = nn.Linear(self.bert_dim, 2* opt.hidden_dim)
            # self.layer_norm = nn.LayerNorm(2* opt.hidden_dim)

            self.sememe_embed = nn.Embedding(opt.sememe_size, opt.sememe_dim, padding_idx=0)
            self.sememe_K = nn.Linear(opt.sememe_dim, 2*opt.hidden_dim)
            self.sememe_Q = nn.Linear(self.bert_dim, 2*opt.hidden_dim)
            self.sememe_V = nn.Linear(self.bert_dim, 2*opt.hidden_dim)

            self.c_drop = nn.Dropout(0.1)
            self.s_drop = nn.Dropout(0.1)
        
        if opt.use_const_and_sememe == "attention2":
            self.const_embed = nn.Embedding(opt.const_size, opt.const_dim, padding_idx=0)
            self.const_pos_embed = nn.Embedding(opt.const_pos_size, opt.const_dim, padding_idx=0)
            self.const_K = nn.Linear(opt.const_dim, 2* opt.hidden_dim)
            self.cs_Q = nn.Linear(self.bert_dim, 2* opt.hidden_dim)
            self.cs_V = nn.Linear(self.bert_dim, 2* opt.hidden_dim)
            # self.layer_norm = nn.LayerNorm(2* opt.hidden_dim)

            self.sememe_embed = nn.Embedding(opt.sememe_size, opt.sememe_dim, padding_idx=0)
            self.sememe_K = nn.Linear(opt.sememe_dim, 2*opt.hidden_dim)

        if opt.use_const_and_sememe == "concat_linear":
            self.const_embed = nn.Embedding(opt.const_size, opt.const_dim, padding_idx=0)
            self.const_pos_embed = nn.Embedding(opt.const_pos_size, opt.const_dim, padding_idx=0)
            self.concat_const = nn.Linear(self.bert_dim+opt.const_dim, self.bert_dim+opt.const_dim)
            self.concat_const_fc = nn.Linear(self.bert_dim+opt.const_dim, 2* opt.hidden_dim)

            self.sememe_embed = nn.Embedding(opt.sememe_size, opt.sememe_dim, padding_idx=0)
            self.concat_sememe = nn.Linear(self.bert_dim + opt.sememe_dim, self.bert_dim + opt.sememe_dim)
            self.concat_sememe_fc = nn.Linear(self.bert_dim + opt.sememe_dim, 2* opt.hidden_dim)
        
        if opt.use_const_and_sememe == "concat_linear2":
            self.const_embed = nn.Embedding(opt.const_size, opt.const_dim, padding_idx=0)
            self.const_pos_embed = nn.Embedding(opt.const_pos_size, opt.const_dim, padding_idx=0)

            self.sememe_embed = nn.Embedding(opt.sememe_size, opt.sememe_dim, padding_idx=0)
            self.concat_cs = nn.Linear(self.bert_dim + opt.sememe_dim + opt.const_dim, self.bert_dim + opt.sememe_dim + opt.const_dim)
            self.concat_cs_fc = nn.Linear(self.bert_dim + opt.sememe_dim + opt.const_dim, 2* opt.hidden_dim)
        
        if opt.use_const_and_sememe == "concat_linear3":
            self.const_embed = nn.Embedding(opt.const_size, opt.const_dim, padding_idx=0)
            self.const_pos_embed = nn.Embedding(opt.const_pos_size, opt.const_dim, padding_idx=0)

            self.sememe_embed = nn.Embedding(opt.sememe_size, opt.sememe_dim, padding_idx=0)
            self.concat_cs = nn.Linear(self.bert_dim + opt.sememe_dim + opt.const_dim, 2* opt.hidden_dim)
        
        if opt.use_const_and_sememe == "attention_pyramid":

            sum_dim = concat_dim = 3 * opt.pyramid_hidden_dim
            self.p1 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim
            self.p2 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim
            self.p3 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim

            self.w0 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w1 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w2 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w3 = nn.Parameter(torch.tensor(1.0)).cuda()
            

            self.const_embed = nn.Embedding(opt.const_size, opt.const_dim, padding_idx=0)
            self.const_pos_embed = nn.Embedding(opt.const_pos_size, opt.const_dim, padding_idx=0)

            self.sememe_embed = nn.Embedding(opt.sememe_size, opt.sememe_dim, padding_idx=0)
            self.concat_cs = nn.Linear(concat_dim, 2* opt.hidden_dim)



            self.const_embed = nn.Embedding(opt.const_size, opt.const_dim, padding_idx=0)
            self.const_pos_embed = nn.Embedding(opt.const_pos_size, opt.const_dim, padding_idx=0)
            self.const_K = nn.Linear(opt.const_dim, opt.pyramid_hidden_dim)
            self.const_Q = nn.Linear(self.bert_dim, opt.pyramid_hidden_dim)
            self.const_V = nn.Linear(self.bert_dim, opt.pyramid_hidden_dim)
            

            self.sememe_embed = nn.Embedding(opt.sememe_size, opt.sememe_dim, padding_idx=0)
            self.sememe_K = nn.Linear(opt.sememe_dim, opt.pyramid_hidden_dim)
            self.sememe_Q = nn.Linear(self.bert_dim, opt.pyramid_hidden_dim)
            self.sememe_V = nn.Linear(self.bert_dim, opt.pyramid_hidden_dim)

            
            self.sc_K = nn.Linear(opt.sememe_dim + opt.const_dim, opt.pyramid_hidden_dim)
            self.sc_Q = nn.Linear(self.bert_dim, opt.pyramid_hidden_dim)
            self.sc_V = nn.Linear(self.bert_dim, opt.pyramid_hidden_dim)

            self.drop = nn.Dropout(0.1)

            self.layer_norm = nn.LayerNorm(2* opt.hidden_dim)
        if opt.use_const_and_sememe == "pyramid":
            sum_dim = self.bert_dim + opt.sememe_dim + opt.const_dim
            concat_dim = sum_dim
            self.p1 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim
            self.p2 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim
            self.p3 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim

            self.w0 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w1 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w2 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w3 = nn.Parameter(torch.tensor(1.0)).cuda()
            

            self.const_embed = nn.Embedding(opt.const_size, opt.const_dim, padding_idx=0)
            self.const_pos_embed = nn.Embedding(opt.const_pos_size, opt.const_dim, padding_idx=0)

            self.sememe_embed = nn.Embedding(opt.sememe_size, opt.sememe_dim, padding_idx=0)
            self.concat_cs = nn.Linear(concat_dim, 2* opt.hidden_dim)

        if opt.use_const_and_sememe == "pyramid2":
            b_s_c_dim = self.bert_dim + opt.sememe_dim + opt.const_dim
            b_s_dim = self.bert_dim + opt.sememe_dim
            b_c_dim = self.bert_dim + opt.const_dim
            s_c_dim = opt.sememe_dim + opt.const_dim
            self.bsc = nn.Linear(b_s_c_dim, opt.pyramid_hidden_dim)
            self.bs = nn.Linear(b_s_dim, opt.pyramid_hidden_dim)
            self.bc = nn.Linear(b_c_dim, opt.pyramid_hidden_dim)
            self.sc = nn.Linear(s_c_dim, opt.pyramid_hidden_dim)

            sum_dim = concat_dim = 4 * opt.pyramid_hidden_dim

            self.p1 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim
            self.p2 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim
            self.p3 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim

            self.w0 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w1 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w2 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w3 = nn.Parameter(torch.tensor(1.0)).cuda()
            

            self.const_embed = nn.Embedding(opt.const_size, opt.const_dim, padding_idx=0)
            self.const_pos_embed = nn.Embedding(opt.const_pos_size, opt.const_dim, padding_idx=0)

            self.sememe_embed = nn.Embedding(opt.sememe_size, opt.sememe_dim, padding_idx=0)
            self.concat_cs = nn.Linear(concat_dim, 2* opt.hidden_dim)
        if opt.use_sememe == "pyramid2_layer_norm_drop":
            b_s_dim = self.bert_dim + opt.sememe_dim
            self.bs = nn.Linear(b_s_dim, opt.pyramid_hidden_dim)

            sum_dim = concat_dim = opt.pyramid_hidden_dim

            self.p1 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim
            self.p2 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim
            self.p3 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim

            self.w0 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w1 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w2 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w3 = nn.Parameter(torch.tensor(1.0)).cuda()
            
            self.sememe_embed = nn.Embedding(opt.sememe_size, opt.sememe_dim, padding_idx=0)

            self.cs_drop = nn.Dropout(opt.cs_dropout)
            self.linear_drop = nn.Dropout(opt.linear_dropout)

            self.concat_cs = nn.Linear(concat_dim, 2* opt.hidden_dim)
            self.layer_norm = nn.LayerNorm(2* opt.hidden_dim)
        if opt.use_const_and_sememe == "pyramid2_layer_norm":
            b_s_c_dim = self.bert_dim + opt.sememe_dim + opt.const_dim
            b_s_dim = self.bert_dim + opt.sememe_dim
            b_c_dim = self.bert_dim + opt.const_dim
            s_c_dim = opt.sememe_dim + opt.const_dim
            self.bsc = nn.Linear(b_s_c_dim, opt.pyramid_hidden_dim)
            self.bs = nn.Linear(b_s_dim, opt.pyramid_hidden_dim)
            self.bc = nn.Linear(b_c_dim, opt.pyramid_hidden_dim)
            self.sc = nn.Linear(s_c_dim, opt.pyramid_hidden_dim)

            sum_dim = concat_dim = 4 * opt.pyramid_hidden_dim

            self.p1 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim
            self.p2 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim
            self.p3 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim

            self.w0 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w1 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w2 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w3 = nn.Parameter(torch.tensor(1.0)).cuda()
            

            self.const_embed = nn.Embedding(opt.const_size, opt.const_dim, padding_idx=0)
            self.const_pos_embed = nn.Embedding(opt.const_pos_size, opt.const_dim, padding_idx=0)

            self.sememe_embed = nn.Embedding(opt.sememe_size, opt.sememe_dim, padding_idx=0)
            self.concat_cs = nn.Linear(concat_dim, 2* opt.hidden_dim)
            self.layer_norm = nn.LayerNorm(2* opt.hidden_dim)
        if opt.use_const_and_sememe == "pyramid2_layer_norm_drop":
            b_s_c_dim = self.bert_dim + opt.sememe_dim + opt.const_dim
            b_s_dim = self.bert_dim + opt.sememe_dim
            b_c_dim = self.bert_dim + opt.const_dim
            s_c_dim = opt.sememe_dim + opt.const_dim
            self.bsc = nn.Linear(b_s_c_dim, opt.pyramid_hidden_dim)
            self.bs = nn.Linear(b_s_dim, opt.pyramid_hidden_dim)
            self.bc = nn.Linear(b_c_dim, opt.pyramid_hidden_dim)
            self.sc = nn.Linear(s_c_dim, opt.pyramid_hidden_dim)

            sum_dim = concat_dim = 4 * opt.pyramid_hidden_dim

            self.p1 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim
            self.p2 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim
            self.p3 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim

            self.w0 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w1 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w2 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w3 = nn.Parameter(torch.tensor(1.0)).cuda()
            

            self.const_embed = nn.Embedding(opt.const_size, opt.const_dim, padding_idx=0)
            self.const_pos_embed = nn.Embedding(opt.const_pos_size, opt.const_dim, padding_idx=0)

            self.sememe_embed = nn.Embedding(opt.sememe_size, opt.sememe_dim, padding_idx=0)

            self.cs_drop = nn.Dropout(opt.cs_dropout)
            self.linear_drop = nn.Dropout(opt.linear_dropout)

            self.concat_cs = nn.Linear(concat_dim, 2* opt.hidden_dim)
            self.layer_norm = nn.LayerNorm(2* opt.hidden_dim)

        if opt.use_const_and_sememe == "pyramid2_layer_norm_FFN":
            b_s_c_dim = self.bert_dim + opt.sememe_dim + opt.const_dim
            b_s_dim = self.bert_dim + opt.sememe_dim
            b_c_dim = self.bert_dim + opt.const_dim
            s_c_dim = opt.sememe_dim + opt.const_dim
            self.bsc = nn.Linear(b_s_c_dim, opt.pyramid_hidden_dim)
            self.bs = nn.Linear(b_s_dim, opt.pyramid_hidden_dim)
            self.bc = nn.Linear(b_c_dim, opt.pyramid_hidden_dim)
            self.sc = nn.Linear(s_c_dim, opt.pyramid_hidden_dim)

            sum_dim = concat_dim = 4 * opt.pyramid_hidden_dim

            self.p1 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim
            self.p2 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim
            self.p3 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim

            self.w0 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w1 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w2 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w3 = nn.Parameter(torch.tensor(1.0)).cuda()
            

            self.const_embed = nn.Embedding(opt.const_size, opt.const_dim, padding_idx=0)
            self.const_pos_embed = nn.Embedding(opt.const_pos_size, opt.const_dim, padding_idx=0)

            self.sememe_embed = nn.Embedding(opt.sememe_size, opt.sememe_dim, padding_idx=0)
            self.concat_cs = nn.Linear(concat_dim, 2* opt.hidden_dim)
            self.layer_norm = nn.LayerNorm(2* opt.hidden_dim)
            self.fc1 = nn.Linear(2* opt.hidden_dim, 4* opt.hidden_dim)
            self.fc2 = nn.Linear(4* opt.hidden_dim, 2* opt.hidden_dim)
            self.drop = nn.Dropout(0.1)

        if opt.use_const_and_sememe == "gate":
            self.const_embed = nn.Embedding(opt.const_size, opt.const_dim, padding_idx=0)
            self.const_pos_embed = nn.Embedding(opt.const_pos_size, opt.const_dim, padding_idx=0)
            self.const_G = nn.Linear(opt.const_dim, 2* opt.hidden_dim)

            self.sememe_embed = nn.Embedding(opt.sememe_size, opt.sememe_dim, padding_idx=0)
            self.sememe_G = nn.Linear(opt.sememe_dim, 2* opt.hidden_dim)
        
                
        if opt.use_const_and_sememe == "gate2":
            self.const_embed = nn.Embedding(opt.const_size, opt.const_dim, padding_idx=0)
            self.const_pos_embed = nn.Embedding(opt.const_pos_size, opt.const_dim, padding_idx=0)

            self.sememe_embed = nn.Embedding(opt.sememe_size, opt.sememe_dim, padding_idx=0)

            self.cs_G = nn.Linear(opt.const_dim + opt.sememe_dim, 2* opt.hidden_dim)

        if opt.use_const_sememe_dep == "pyramid2_layer_norm_drop":
            b_s_c_d_dim = self.bert_dim + opt.sememe_dim + opt.const_dim + opt.dep_dim
            b_s_dim = self.bert_dim + opt.sememe_dim
            b_c_dim = self.bert_dim + opt.const_dim
            b_d_dim = self.bert_dim + opt.dep_dim
            s_c_d_dim = opt.sememe_dim + opt.const_dim + opt.dep_dim

            self.bscd = nn.Linear(b_s_c_d_dim, opt.pyramid_hidden_dim)
            self.bs = nn.Linear(b_s_dim, opt.pyramid_hidden_dim)
            self.bc = nn.Linear(b_c_dim, opt.pyramid_hidden_dim)
            self.bd = nn.Linear(b_d_dim, opt.pyramid_hidden_dim)
            self.scd = nn.Linear(s_c_d_dim, opt.pyramid_hidden_dim)

            sum_dim = concat_dim = 5 * opt.pyramid_hidden_dim

            self.p1 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim
            self.p2 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim
            self.p3 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim

            self.w0 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w1 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w2 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w3 = nn.Parameter(torch.tensor(1.0)).cuda()


            self.const_embed = nn.Embedding(opt.const_size, opt.const_dim, padding_idx=0)
            self.const_pos_embed = nn.Embedding(opt.const_pos_size, opt.const_dim, padding_idx=0)

            self.sememe_embed = nn.Embedding(opt.sememe_size, opt.sememe_dim, padding_idx=0)

            self.dep_embed = nn.Embedding(opt.dep_size, opt.dep_dim, padding_idx=0)
            self.dep_pos_embed = nn.Embedding(opt.dep_pos_size, opt.dep_dim, padding_idx=0)

            self.cs_drop = nn.Dropout(opt.cs_dropout)
            self.linear_drop = nn.Dropout(opt.linear_dropout)

            self.concat_cs = nn.Linear(concat_dim, 2* opt.hidden_dim)
            self.layer_norm = nn.LayerNorm(2* opt.hidden_dim)
        if opt.use_const_and_dep == "pyramid2_layer_norm_drop":
            b_c_d_dim = self.bert_dim + opt.const_dim + opt.dep_dim
            b_c_dim = self.bert_dim + opt.const_dim
            b_d_dim = self.bert_dim + opt.dep_dim
            c_d_dim = opt.const_dim + opt.dep_dim

            self.bcd = nn.Linear(b_c_d_dim, opt.pyramid_hidden_dim)
            self.bc = nn.Linear(b_c_dim, opt.pyramid_hidden_dim)
            self.bd = nn.Linear(b_d_dim, opt.pyramid_hidden_dim)
            self.cd = nn.Linear(c_d_dim, opt.pyramid_hidden_dim)

            sum_dim = concat_dim = 4 * opt.pyramid_hidden_dim

            self.p1 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim
            self.p2 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim
            self.p3 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim

            self.w0 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w1 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w2 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w3 = nn.Parameter(torch.tensor(1.0)).cuda()


            self.const_embed = nn.Embedding(opt.const_size, opt.const_dim, padding_idx=0)
            self.const_pos_embed = nn.Embedding(opt.const_pos_size, opt.const_dim, padding_idx=0)

            self.dep_embed = nn.Embedding(opt.dep_size, opt.dep_dim, padding_idx=0)
            self.dep_pos_embed = nn.Embedding(opt.dep_pos_size, opt.dep_dim, padding_idx=0)

            self.cs_drop = nn.Dropout(opt.cs_dropout)
            self.linear_drop = nn.Dropout(opt.linear_dropout)

            self.concat_cs = nn.Linear(concat_dim, 2* opt.hidden_dim)
            self.layer_norm = nn.LayerNorm(2* opt.hidden_dim)

        if opt.use_sememe_and_dep == "pyramid2_layer_norm_drop":
            b_s_d_dim = self.bert_dim + opt.sememe_dim + opt.dep_dim
            b_s_dim = self.bert_dim + opt.sememe_dim
            b_d_dim = self.bert_dim + opt.dep_dim
            s_d_dim = opt.sememe_dim + opt.dep_dim

            self.bsd = nn.Linear(b_s_d_dim, opt.pyramid_hidden_dim)
            self.bs = nn.Linear(b_s_dim, opt.pyramid_hidden_dim)
            self.bd = nn.Linear(b_d_dim, opt.pyramid_hidden_dim)
            self.sd = nn.Linear(s_d_dim, opt.pyramid_hidden_dim)

            sum_dim = concat_dim = 4 * opt.pyramid_hidden_dim

            self.p1 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim
            self.p2 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim
            self.p3 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim

            self.w0 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w1 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w2 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w3 = nn.Parameter(torch.tensor(1.0)).cuda()

            self.sememe_embed = nn.Embedding(opt.sememe_size, opt.sememe_dim, padding_idx=0)

            self.dep_embed = nn.Embedding(opt.dep_size, opt.dep_dim, padding_idx=0)
            self.dep_pos_embed = nn.Embedding(opt.dep_pos_size, opt.dep_dim, padding_idx=0)

            self.cs_drop = nn.Dropout(opt.cs_dropout)
            self.linear_drop = nn.Dropout(opt.linear_dropout)

            self.concat_cs = nn.Linear(concat_dim, 2* opt.hidden_dim)
            self.layer_norm = nn.LayerNorm(2* opt.hidden_dim)

        if opt.use_const_sememe_dep == "attention":

            self.const_K = nn.Linear(opt.const_dim, 2* opt.hidden_dim)
            self.const_Q = nn.Linear(self.bert_dim, 2* opt.hidden_dim)
            self.const_V = nn.Linear(self.bert_dim, 2* opt.hidden_dim)

            self.sememe_K = nn.Linear(opt.sememe_dim, 2*opt.hidden_dim)
            self.sememe_Q = nn.Linear(self.bert_dim, 2*opt.hidden_dim)
            self.sememe_V = nn.Linear(self.bert_dim, 2*opt.hidden_dim)

            self.dep_K = nn.Linear(opt.dep_dim, 2*opt.hidden_dim)
            self.dep_Q = nn.Linear(self.bert_dim, 2*opt.hidden_dim)
            self.dep_V = nn.Linear(self.bert_dim, 2*opt.hidden_dim)

            self.const_embed = nn.Embedding(opt.const_size, opt.const_dim, padding_idx=0)
            self.const_pos_embed = nn.Embedding(opt.const_pos_size, opt.const_dim, padding_idx=0)

            self.sememe_embed = nn.Embedding(opt.sememe_size, opt.sememe_dim, padding_idx=0)

            self.dep_embed = nn.Embedding(opt.dep_size, opt.dep_dim, padding_idx=0)
            self.dep_pos_embed = nn.Embedding(opt.dep_pos_size, opt.dep_dim, padding_idx=0)

            self.cs_drop = nn.Dropout(opt.cs_dropout)
            self.att_drop = nn.Dropout(opt.linear_dropout)

            self.layer_norm = nn.LayerNorm(2* opt.hidden_dim)
        if opt.use_const_sememe_dep == "pyramid2_LN_D":
            b_s_c_d_dim = self.bert_dim + opt.sememe_dim + opt.const_dim + opt.dep_dim
            b_s_dim = self.bert_dim + opt.sememe_dim
            b_c_dim = self.bert_dim + opt.const_dim
            b_d_dim = self.bert_dim + opt.dep_dim

            self.bscd = nn.Linear(b_s_c_d_dim, opt.pyramid_hidden_dim)
            self.bs = nn.Linear(b_s_dim, opt.pyramid_hidden_dim)
            self.bc = nn.Linear(b_c_dim, opt.pyramid_hidden_dim)
            self.bd = nn.Linear(b_d_dim, opt.pyramid_hidden_dim)

            sum_dim = concat_dim = 4 * opt.pyramid_hidden_dim

            self.p1 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim
            self.p2 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim
            self.p3 = nn.Linear(sum_dim, sum_dim // 2)
            sum_dim = sum_dim // 2
            concat_dim += sum_dim

            self.w0 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w1 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w2 = nn.Parameter(torch.tensor(1.0)).cuda()
            self.w3 = nn.Parameter(torch.tensor(1.0)).cuda()


            self.const_embed = nn.Embedding(opt.const_size, opt.const_dim, padding_idx=0)
            self.const_pos_embed = nn.Embedding(opt.const_pos_size, opt.const_dim, padding_idx=0)

            self.sememe_embed = nn.Embedding(opt.sememe_size, opt.sememe_dim, padding_idx=0)

            self.dep_embed = nn.Embedding(opt.dep_size, opt.dep_dim, padding_idx=0)
            self.dep_pos_embed = nn.Embedding(opt.dep_pos_size, opt.dep_dim, padding_idx=0)

            self.cs_drop = nn.Dropout(opt.cs_dropout)
            self.linear_drop = nn.Dropout(opt.linear_dropout)

            self.concat_cs = nn.Linear(concat_dim, 2* opt.hidden_dim)
            self.layer_norm = nn.LayerNorm(2* opt.hidden_dim)
        if opt.use_const_sememe_dep == "gate":
            self.const_embed = nn.Embedding(opt.const_size, opt.const_dim, padding_idx=0)
            self.const_pos_embed = nn.Embedding(opt.const_pos_size, opt.const_dim, padding_idx=0)
            self.const_G = nn.Linear(opt.const_dim, 2* opt.hidden_dim)

            self.sememe_embed = nn.Embedding(opt.sememe_size, opt.sememe_dim, padding_idx=0)
            self.sememe_G = nn.Linear(opt.sememe_dim, 2* opt.hidden_dim)

            self.dep_embed = nn.Embedding(opt.dep_size, opt.dep_dim, padding_idx=0)
            self.dep_pos_embed = nn.Embedding(opt.dep_pos_size, opt.dep_dim, padding_idx=0)
            self.dep_G = nn.Linear(opt.dep_dim, 2* opt.hidden_dim)

            self.cs_drop = nn.Dropout(opt.cs_dropout)
            self.gate_drop = nn.Dropout(opt.linear_dropout)

            self.layer_norm = nn.LayerNorm(2* opt.hidden_dim)

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
    
    def forward(self, inputs, labels = None,  debugger=None, temperature=None, const=None, const_pos=None):
        self.count += 1
        #self.debug_scalar_mix()
        #sys.exit(0)
        text_indices, aspect_indices, aspect_bert_indices, left_indices, left_bert_indices, adj, pos_indices, rel_indices, \
        text_bert_indices, text_raw_bert_indices, bert_segments_ids, bert_token_masks, word_seq_lengths, words, \
        aux_aspect_targets, const, const_pos, sememe, dep, dep_pos = inputs
        
        # adj = torch.zeros((text_indices.shape[0], text_indices.shape[1], text_indices.shape[1]))

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
        self.nt_xent_criterion = NTXentLoss(self.opt.device, self.opt.batch_size, 1.0, True)
        
         
        
        for b in range(batch_size):
            start_len = 1 # excluding cls
            #print(words[b], word_seq_lengths[b])
            assert len(words[b]) == len(word_seq_lengths[b])
            for i in range(len(word_seq_lengths[b])):
                merged_layer[b, i, :] = torch.mean(encoded_layer[b, start_len:start_len + word_seq_lengths[b][i], :], dim=0).squeeze(dim=0)
                start_len += word_seq_lengths[b][i]
        text = None
        # bert编码后句子中每个词的向量表示
        if self.opt.use_const == "add":
            const_tag_embed = self.const_embed(const)
            const_pos_embed = self.const_pos_embed(const_pos)
            const_info_embed = torch.sum(const_tag_embed*const_pos_embed, dim=2)
            const_w = self.const_W(const_info_embed)
            context_w = self.context_W(merged_layer)
            text = self.const_fc(self.add_activation(const_w) + self.add_activation(context_w))
            text = self.add_activation(text)
            text = text + self.bert_linear(merged_layer)
        elif self.opt.use_const == "concat_linear":
            const_tag_embed = self.const_embed(const)
            const_pos_embed = self.const_pos_embed(const_pos)
            const_info_embed = torch.sum(const_tag_embed*const_pos_embed, dim=2)
            add_const = torch.concat([const_info_embed, merged_layer], dim=2)
            text = self.concat_const(add_const)
            text = torch.sigmoid(text)
            text = self.concat_const_fc(text)
            text = torch.sigmoid(text)
            text = text + self.bert_linear(merged_layer)
        elif self.opt.use_const == "attention":
            const_tag_embed = self.const_embed(const)
            const_pos_embed = self.const_pos_embed(const_pos)
            const_info_embed = torch.sum(const_tag_embed*const_pos_embed, dim=2)
            Q = self.const_Q(merged_layer)
            K = self.const_K(const_info_embed)
            V = self.const_V(merged_layer)
            attn = torch.matmul(Q, K.transpose(-2, -1))
            p_attn = F.softmax(attn, dim=-1)
            text = torch.matmul(p_attn, V)
            text = text + self.bert_linear(merged_layer)
            # text = self.layer_norm(text)
        elif self.opt.use_const == "concat_attention":
            const_tag_embed = self.const_embed(const)
            const_pos_embed = self.const_pos_embed(const_pos)
            const_info_embed = torch.sum(const_tag_embed*const_pos_embed, dim=2)
            concat_const = torch.concat([merged_layer, const_info_embed], dim=2)
            Q = self.const_Q(concat_const)
            K = self.const_K(const_info_embed)
            V = self.const_V(concat_const)
            attn = torch.matmul(Q, K.transpose(-2, -1))
            p_attn = F.softmax(attn, dim=-1)
            text = torch.matmul(p_attn, V)
            text = text + self.bert_linear(merged_layer)
            # text = self.layer_norm(text)
        elif self.opt.use_const == "concat_gate_attention":
            const_tag_embed = self.const_embed(const)
            const_pos_embed = self.const_pos_embed(const_pos)
            const_info_embed = torch.sum(const_tag_embed*const_pos_embed, dim=2)
            concat_const = torch.concat([merged_layer, const_info_embed], dim=2)
            raw_text = self.bert_linear(merged_layer)
            const_g = self.gate_activation(self.const_G(const_info_embed)) * raw_text
            Q = self.const_Q(const_g)
            K = self.const_K(const_info_embed)
            V = self.const_V(concat_const)
            attn = torch.matmul(Q, K.transpose(-2, -1))
            p_attn = F.softmax(attn, dim=-1)
            text = torch.matmul(p_attn, V)
            text = text + raw_text
            # text = self.layer_norm(text)
        elif self.opt.use_const == "gate":
            const_tag_embed = self.const_embed(const)
            const_pos_embed = self.const_pos_embed(const_pos)
            const_info_embed = torch.sum(const_tag_embed * const_pos_embed, dim=2)
            raw_text = self.bert_linear(merged_layer)
            const_g = self.gate_activation(self.const_G(const_info_embed)) * raw_text
            text = const_g + raw_text
            # text = self.layer_norm(text)
        elif self.opt.use_const == "pyramid2_layer_norm_drop":
            const_tag_embed = self.const_embed(const)
            const_pos_embed = self.const_pos_embed(const_pos)
            const_info_embed = torch.sum(const_tag_embed * const_pos_embed, dim=2)
            const_info_embed = self.cs_drop(const_info_embed)
            bc_text = self.linear_drop(torch.sigmoid(self.bc(torch.cat([merged_layer, const_info_embed], dim=-1))))

            cs_text_0 = bc_text
            cs_text_1 = self.linear_drop(torch.tanh(self.p1(cs_text_0)))
            cs_text_2 = self.linear_drop(torch.tanh(self.p2(cs_text_1)))
            cs_text_3 = self.linear_drop(torch.tanh(self.p3(cs_text_2)))
            concat_text = torch.concat([cs_text_0 * self.w0, cs_text_1 * self.w1, cs_text_2 * self.w2, cs_text_3 * self.w3], dim=2)
            text = self.linear_drop(torch.tanh(self.concat_cs(concat_text)))
            text = self.layer_norm(text) + self.bert_linear(merged_layer)
            # print(self.w0, self.w1, self.w2, self.w3)
            # print(self.p1.weight)
        if self.opt.use_dep == "pyramid2_layer_norm_drop":
            dep_tag_embed = self.dep_embed(dep)
            dep_pos_embed = self.dep_pos_embed(dep_pos)
            dep_info_embed = torch.sum(dep_tag_embed * dep_pos_embed, dim=2)
            dep_info_embed = self.cs_drop(dep_info_embed)
            bd_text = self.linear_drop(torch.sigmoid(self.bd(torch.cat([merged_layer, dep_info_embed], dim=-1))))

            cs_text_0 = bd_text
            cs_text_1 = self.linear_drop(torch.tanh(self.p1(cs_text_0)))
            cs_text_2 = self.linear_drop(torch.tanh(self.p2(cs_text_1)))
            cs_text_3 = self.linear_drop(torch.tanh(self.p3(cs_text_2)))
            concat_text = torch.concat([cs_text_0 * self.w0, cs_text_1 * self.w1, cs_text_2 * self.w2, cs_text_3 * self.w3], dim=2)
            text = self.linear_drop(torch.tanh(self.concat_cs(concat_text)))
            text = self.layer_norm(text) + self.bert_linear(merged_layer)

        if self.opt.use_sememe == "concat_linear":
            sememe_embed = self.sememe_embed(sememe)
            sememe_embed = torch.mean(sememe_embed, dim=2)
            add_sememe = torch.concat([merged_layer, sememe_embed], dim=2)
            text = self.concat_sememe(add_sememe)
            text = torch.sigmoid(text)
            text = self.concat_sememe_fc(text)
            text = torch.sigmoid(text)
            text = text + self.bert_linear(merged_layer)
        elif self.opt.use_sememe == "attention":
            sememe_embed = self.sememe_embed(sememe)
            sememe_embed = torch.mean(sememe_embed, dim=2)
            Q = self.sememe_Q(merged_layer)
            K = self.sememe_K(sememe_embed)
            # K = self.sememe_w2(self.sememe_drop(F.relu(self.sememe_w1(K))))
            V = self.sememe_V(merged_layer)
            attn = torch.matmul(Q, K.transpose(-2, -1))
            p_attn = F.softmax(attn, dim=-1)
            text = torch.matmul(p_attn, V)
            text = text + self.bert_linear(merged_layer)
            # text = self.layer_norm(text)
        elif self.opt.use_sememe == "gate":
            sememe_embed = self.sememe_embed(sememe)
            sememe_embed = torch.mean(sememe_embed, dim=2)
            raw_text = self.bert_linear(merged_layer)
            sememe_g = self.gate_activation(self.sememe_G(sememe_embed)) * raw_text
            text = sememe_g + raw_text
            # text = self.layer_norm(text)
        elif self.opt.use_sememe == "pyramid2_layer_norm_drop":
            sememe_embed = self.sememe_embed(sememe)
            sememe_embed = torch.mean(sememe_embed, dim=2)
            sememe_embed = self.cs_drop(sememe_embed)


            bs_text = self.linear_drop(torch.sigmoid(self.bs(torch.cat([merged_layer, sememe_embed], dim=-1))))

            cs_text_0 = bs_text
            cs_text_1 = self.linear_drop(torch.tanh(self.p1(cs_text_0)))
            cs_text_2 = self.linear_drop(torch.tanh(self.p2(cs_text_1)))
            cs_text_3 = self.linear_drop(torch.tanh(self.p3(cs_text_2)))
            concat_text = torch.concat([cs_text_0 * self.w0, cs_text_1 * self.w1, cs_text_2 * self.w2, cs_text_3 * self.w3], dim=2)
            text = self.linear_drop(torch.tanh(self.concat_cs(concat_text)))
            text = self.layer_norm(text) + self.bert_linear(merged_layer)
            # print(self.w0, self.w1, self.w2, self.w3)
            # print(self.p1.weight)

        
        if self.opt.use_const_and_sememe == "attention":
            const_tag_embed = self.const_embed(const)
            const_pos_embed = self.const_pos_embed(const_pos)
            const_info_embed = torch.sum(const_tag_embed*const_pos_embed, dim=2)
            Q = self.const_Q(merged_layer)
            K = self.const_K(const_info_embed)
            V = self.const_V(merged_layer)
            attn = torch.matmul(Q, K.transpose(-2, -1))
            p_attn = F.softmax(attn, dim=-1)
            const_text = torch.matmul(p_attn, V)

            sememe_embed = self.sememe_embed(sememe)
            sememe_embed = torch.mean(sememe_embed, dim=2)
            Q = self.sememe_Q(merged_layer)
            K = self.sememe_K(sememe_embed)
            V = self.sememe_V(merged_layer)
            attn = torch.matmul(Q, K.transpose(-2, -1))
            p_attn = F.softmax(attn, dim=-1)
            sememe_text = torch.matmul(p_attn, V)
            text = const_text + self.bert_linear(merged_layer) + sememe_text
        elif self.opt.use_const_and_sememe == "attention_FFN":
            const_tag_embed = self.const_embed(const)
            const_pos_embed = self.const_pos_embed(const_pos)
            const_info_embed = torch.sum(const_tag_embed*const_pos_embed, dim=2)
            Q = self.const_Q(merged_layer)
            K = self.const_K(const_info_embed)
            V = self.const_V(merged_layer)
            attn = torch.matmul(Q, K.transpose(-2, -1))
            p_attn = F.softmax(attn, dim=-1)
            p_attn = self.drop(p_attn)
            const_text = torch.matmul(p_attn, V)

            sememe_embed = self.sememe_embed(sememe)
            sememe_embed = torch.mean(sememe_embed, dim=2)
            Q = self.sememe_Q(merged_layer)
            K = self.sememe_K(sememe_embed)
            V = self.sememe_V(merged_layer)
            attn = torch.matmul(Q, K.transpose(-2, -1))
            p_attn = F.softmax(attn, dim=-1)
            p_attn = self.drop(p_attn)
            sememe_text = torch.matmul(p_attn, V)

            sc_embed = torch.cat([const_info_embed, sememe_embed], dim=-1)
            Q = self.sc_Q(merged_layer)
            K = self.sc_K(sc_embed)
            V = self.sc_V(merged_layer)
            attn = torch.matmul(Q, K.transpose(-2, -1))
            p_attn = F.softmax(attn, dim=-1)
            p_attn = self.drop(p_attn)
            sc_text = torch.matmul(p_attn, V)

            text = self.fc2(self.drop(F.relu(self.fc1(torch.cat([const_text, sememe_text, sc_text], dim=-1)))))
            text = self.layer_norm(text) + self.bert_linear(merged_layer)
        elif self.opt.use_const_and_sememe == "attention_add":
            const_tag_embed = self.const_embed(const)
            const_pos_embed = self.const_pos_embed(const_pos)
            const_info_embed = torch.sum(const_tag_embed*const_pos_embed, dim=2)
            Q = self.const_Q(merged_layer)
            K = self.const_K(const_info_embed)
            V = self.const_V(merged_layer)
            attn = torch.matmul(Q, K.transpose(-2, -1))
            p_attn = F.softmax(attn, dim=-1)
            p_attn = self.drop(p_attn)
            const_text = torch.matmul(p_attn, V)

            sememe_embed = self.sememe_embed(sememe)
            sememe_embed = torch.mean(sememe_embed, dim=2)
            Q = self.sememe_Q(merged_layer)
            K = self.sememe_K(sememe_embed)
            V = self.sememe_V(merged_layer)
            attn = torch.matmul(Q, K.transpose(-2, -1))
            p_attn = F.softmax(attn, dim=-1)
            p_attn = self.drop(p_attn)
            sememe_text = torch.matmul(p_attn, V)

            sc_embed = torch.cat([const_info_embed, sememe_embed], dim=-1)
            Q = self.sc_Q(merged_layer)
            K = self.sc_K(sc_embed)
            V = self.sc_V(merged_layer)
            attn = torch.matmul(Q, K.transpose(-2, -1))
            p_attn = F.softmax(attn, dim=-1)
            p_attn = self.drop(p_attn)
            sc_text = torch.matmul(p_attn, V)
            text = self.layer_norm(const_text+sememe_text+sc_text) + self.bert_linear(merged_layer)
        elif self.opt.use_const_and_sememe == "attention_pyramid":
            const_tag_embed = self.const_embed(const)
            const_pos_embed = self.const_pos_embed(const_pos)
            const_info_embed = torch.sum(const_tag_embed*const_pos_embed, dim=2)
            Q = self.const_Q(merged_layer)
            K = self.const_K(const_info_embed)
            V = self.const_V(merged_layer)
            attn = torch.matmul(Q, K.transpose(-2, -1))
            p_attn = F.softmax(attn, dim=-1)
            p_attn = self.drop(p_attn)
            const_text = torch.matmul(p_attn, V)

            sememe_embed = self.sememe_embed(sememe)
            sememe_embed = torch.mean(sememe_embed, dim=2)
            Q = self.sememe_Q(merged_layer)
            K = self.sememe_K(sememe_embed)
            V = self.sememe_V(merged_layer)
            attn = torch.matmul(Q, K.transpose(-2, -1))
            p_attn = F.softmax(attn, dim=-1)
            p_attn = self.drop(p_attn)
            sememe_text = torch.matmul(p_attn, V)

            sc_embed = torch.cat([const_info_embed, sememe_embed], dim=-1)
            Q = self.sc_Q(merged_layer)
            K = self.sc_K(sc_embed)
            V = self.sc_V(merged_layer)
            attn = torch.matmul(Q, K.transpose(-2, -1))
            p_attn = F.softmax(attn, dim=-1)
            p_attn = self.drop(p_attn)
            sc_text = torch.matmul(p_attn, V)

            cs_text_0 = torch.concat([const_text, sememe_text, sc_text], dim=2)
            cs_text_1 = torch.tanh(self.p1(cs_text_0))
            cs_text_2 = torch.tanh(self.p2(cs_text_1))
            cs_text_3 = torch.tanh(self.p3(cs_text_2))
            concat_text = torch.concat([cs_text_0 * self.w0, cs_text_1 * self.w1, cs_text_2 * self.w2, cs_text_3 * self.w3], dim=2)
            text = torch.tanh(self.concat_cs(concat_text))
            text = self.layer_norm(text) + self.bert_linear(merged_layer)
        elif self.opt.use_const_and_sememe == "attention_drop":
            const_tag_embed = self.const_embed(const)
            const_pos_embed = self.const_pos_embed(const_pos)
            const_info_embed = torch.sum(const_tag_embed*const_pos_embed, dim=2)
            Q = self.const_Q(merged_layer)
            K = self.const_K(const_info_embed)
            V = self.const_V(merged_layer)
            attn = torch.matmul(Q, K.transpose(-2, -1))
            p_attn = self.c_drop(F.softmax(attn, dim=-1))
            const_text = torch.matmul(p_attn, V)

            sememe_embed = self.sememe_embed(sememe)
            sememe_embed = torch.mean(sememe_embed, dim=2)
            Q = self.sememe_Q(merged_layer)
            K = self.sememe_K(sememe_embed)
            V = self.sememe_V(merged_layer)
            attn = torch.matmul(Q, K.transpose(-2, -1))
            p_attn = self.s_drop(F.softmax(attn, dim=-1))
            sememe_text = torch.matmul(p_attn, V)
            text = const_text + self.bert_linear(merged_layer) + sememe_text


        elif self.opt.use_const_and_sememe == "attention2":
            const_tag_embed = self.const_embed(const)
            const_pos_embed = self.const_pos_embed(const_pos)
            const_info_embed = torch.sum(const_tag_embed*const_pos_embed, dim=2)
            Q = self.cs_Q(merged_layer)
            c_K = self.const_K(const_info_embed)
            V = self.cs_V(merged_layer)
            c_attn = torch.matmul(Q, c_K.transpose(-2, -1))
            c_p_attn = F.softmax(c_attn, dim=-1)
            const_text = torch.matmul(c_p_attn, V)

            sememe_embed = self.sememe_embed(sememe)
            sememe_embed = torch.mean(sememe_embed, dim=2)
            s_K = self.sememe_K(sememe_embed)
            s_attn = torch.matmul(Q, s_K.transpose(-2, -1))
            s_p_attn = F.softmax(s_attn, dim=-1)
            sememe_text = torch.matmul(s_p_attn, V)

            text = const_text + self.bert_linear(merged_layer) + sememe_text
            
        elif self.opt.use_const_and_sememe == "concat_linear":
            const_tag_embed = self.const_embed(const)
            const_pos_embed = self.const_pos_embed(const_pos)
            const_info_embed = torch.sum(const_tag_embed*const_pos_embed, dim=2)
            add_const = torch.concat([const_info_embed, merged_layer], dim=2)
            const_text = self.concat_const(add_const)
            const_text = torch.sigmoid(const_text)
            const_text = self.concat_const_fc(const_text)
            const_text = torch.sigmoid(const_text)
            
            sememe_embed = self.sememe_embed(sememe)
            sememe_embed = torch.mean(sememe_embed, dim=2)
            add_sememe = torch.concat([merged_layer, sememe_embed], dim=2)
            sememe_text = self.concat_sememe(add_sememe)
            sememe_text = torch.sigmoid(sememe_text)
            sememe_text = self.concat_sememe_fc(sememe_text)
            sememe_text = torch.sigmoid(sememe_text)

            text = const_text + sememe_text + self.bert_linear(merged_layer)
        elif self.opt.use_const_and_sememe == "concat_linear2":
            const_tag_embed = self.const_embed(const)
            const_pos_embed = self.const_pos_embed(const_pos)
            const_info_embed = torch.sum(const_tag_embed*const_pos_embed, dim=2)
            add_const = torch.concat([const_info_embed, merged_layer], dim=2)

            sememe_embed = self.sememe_embed(sememe)
            sememe_embed = torch.mean(sememe_embed, dim=2)
            add_const_sememe = torch.concat([sememe_embed, add_const], dim=2)

            cs_text = self.concat_cs(add_const_sememe)
            cs_text = torch.sigmoid(cs_text)
            cs_text = self.concat_cs_fc(cs_text)
            cs_text = torch.sigmoid(cs_text)

        elif self.opt.use_const_and_sememe == "concat_linear3":
            const_tag_embed = self.const_embed(const)
            const_pos_embed = self.const_pos_embed(const_pos)
            const_info_embed = torch.sum(const_tag_embed*const_pos_embed, dim=2)
            add_const = torch.concat([const_info_embed, merged_layer], dim=2)

            sememe_embed = self.sememe_embed(sememe)
            sememe_embed = torch.mean(sememe_embed, dim=2)
            add_const_sememe = torch.concat([sememe_embed, add_const], dim=2)

            cs_text = self.concat_cs(add_const_sememe)
            cs_text = torch.tanh(cs_text)
            text = cs_text + self.bert_linear(merged_layer)

        elif self.opt.use_const_and_sememe == "gate":
            const_tag_embed = self.const_embed(const)
            const_pos_embed = self.const_pos_embed(const_pos)
            const_info_embed = torch.sum(const_tag_embed * const_pos_embed, dim=2)
            raw_text = self.bert_linear(merged_layer)
            const_g = self.gate_activation(self.const_G(const_info_embed)) * raw_text
            sememe_embed = self.sememe_embed(sememe)
            sememe_embed = torch.mean(sememe_embed, dim=2)
            sememe_g = self.gate_activation(self.sememe_G(sememe_embed)) * raw_text
            text = sememe_g + raw_text + const_g
        elif self.opt.use_const_and_sememe == "gate2":
            const_tag_embed = self.const_embed(const)
            const_pos_embed = self.const_pos_embed(const_pos)
            const_info_embed = torch.sum(const_tag_embed * const_pos_embed, dim=2)

            sememe_embed = self.sememe_embed(sememe)
            sememe_embed = torch.mean(sememe_embed, dim=2)
            raw_text = self.bert_linear(merged_layer)
            const_sememe = torch.concat([const_info_embed, sememe_embed], dim=2)
            cs_g = self.gate_activation(self.cs_G(const_sememe)) * raw_text
            text = raw_text + cs_g


        elif self.opt.use_const_and_sememe == "pyramid":
            const_tag_embed = self.const_embed(const)
            const_pos_embed = self.const_pos_embed(const_pos)
            const_info_embed = torch.sum(const_tag_embed * const_pos_embed, dim=2)

            sememe_embed = self.sememe_embed(sememe)
            sememe_embed = torch.mean(sememe_embed, dim=2)
            cs_text_0 = torch.concat([const_info_embed, sememe_embed, merged_layer], dim=2)
            cs_text_1 = torch.tanh(self.p1(cs_text_0))
            cs_text_2 = torch.tanh(self.p2(cs_text_1))
            cs_text_3 = torch.tanh(self.p3(cs_text_2))
            concat_text = torch.concat([cs_text_0 * self.w0, cs_text_1 * self.w1, cs_text_2 * self.w2, cs_text_3 * self.w3], dim=2)
            text = torch.tanh(self.concat_cs(concat_text))
            text = text + self.bert_linear(merged_layer)

        elif self.opt.use_const_and_sememe == "pyramid2":
            const_tag_embed = self.const_embed(const)
            const_pos_embed = self.const_pos_embed(const_pos)
            const_info_embed = torch.sum(const_tag_embed * const_pos_embed, dim=2)

            sememe_embed = self.sememe_embed(sememe)
            sememe_embed = torch.mean(sememe_embed, dim=2)
            bsc_text = torch.sigmoid(self.bsc(torch.cat([merged_layer, const_info_embed, sememe_embed], dim=-1)))
            bs_text = torch.sigmoid(self.bs(torch.cat([merged_layer, sememe_embed], dim=-1)))
            bc_text = torch.sigmoid(self.bc(torch.cat([merged_layer, const_info_embed], dim=-1)))
            sc_text = torch.sigmoid(self.sc(torch.cat([sememe_embed, const_info_embed], dim=-1)))

            cs_text_0 = torch.concat([bsc_text, bs_text, bc_text, sc_text], dim=2)
            cs_text_1 = torch.tanh(self.p1(cs_text_0))
            cs_text_2 = torch.tanh(self.p2(cs_text_1))
            cs_text_3 = torch.tanh(self.p3(cs_text_2))
            concat_text = torch.concat([cs_text_0 * self.w0, cs_text_1 * self.w1, cs_text_2 * self.w2, cs_text_3 * self.w3], dim=2)
            text = torch.tanh(self.concat_cs(concat_text))
            text = text + self.bert_linear(merged_layer)

        elif self.opt.use_const_and_sememe == "pyramid2_layer_norm":
            const_tag_embed = self.const_embed(const)
            const_pos_embed = self.const_pos_embed(const_pos)
            const_info_embed = torch.sum(const_tag_embed * const_pos_embed, dim=2)

            sememe_embed = self.sememe_embed(sememe)
            sememe_embed = torch.mean(sememe_embed, dim=2)
            bsc_text = torch.sigmoid(self.bsc(torch.cat([merged_layer, const_info_embed, sememe_embed], dim=-1)))
            bs_text = torch.sigmoid(self.bs(torch.cat([merged_layer, sememe_embed], dim=-1)))
            bc_text = torch.sigmoid(self.bc(torch.cat([merged_layer, const_info_embed], dim=-1)))
            sc_text = torch.sigmoid(self.sc(torch.cat([sememe_embed, const_info_embed], dim=-1)))

            cs_text_0 = torch.concat([bsc_text, bs_text, bc_text, sc_text], dim=2)
            cs_text_1 = torch.tanh(self.p1(cs_text_0))
            cs_text_2 = torch.tanh(self.p2(cs_text_1))
            cs_text_3 = torch.tanh(self.p3(cs_text_2))
            concat_text = torch.concat([cs_text_0 * self.w0, cs_text_1 * self.w1, cs_text_2 * self.w2, cs_text_3 * self.w3], dim=2)
            text = torch.tanh(self.concat_cs(concat_text))
            text = self.layer_norm(text) + self.bert_linear(merged_layer)

        elif self.opt.use_const_and_sememe == "pyramid2_layer_norm_drop":
            const_tag_embed = self.const_embed(const)
            const_pos_embed = self.const_pos_embed(const_pos)
            const_info_embed = torch.sum(const_tag_embed * const_pos_embed, dim=2)
            const_info_embed = self.cs_drop(const_info_embed)

            sememe_embed = self.sememe_embed(sememe)
            sememe_embed = torch.mean(sememe_embed, dim=2)
            sememe_embed = self.cs_drop(sememe_embed)

            bsc_text = self.linear_drop(torch.sigmoid(self.bsc(torch.cat([merged_layer, const_info_embed, sememe_embed], dim=-1))))
            bs_text = self.linear_drop(torch.sigmoid(self.bs(torch.cat([merged_layer, sememe_embed], dim=-1))))
            bc_text = self.linear_drop(torch.sigmoid(self.bc(torch.cat([merged_layer, const_info_embed], dim=-1))))
            sc_text = self.linear_drop(torch.sigmoid(self.sc(torch.cat([sememe_embed, const_info_embed], dim=-1))))

            cs_text_0 = torch.concat([bsc_text, bs_text, bc_text, sc_text], dim=2)
            cs_text_1 = self.linear_drop(torch.tanh(self.p1(cs_text_0)))
            cs_text_2 = self.linear_drop(torch.tanh(self.p2(cs_text_1)))
            cs_text_3 = self.linear_drop(torch.tanh(self.p3(cs_text_2)))
            concat_text = torch.concat([cs_text_0 * self.w0, cs_text_1 * self.w1, cs_text_2 * self.w2, cs_text_3 * self.w3], dim=2)
            text = self.linear_drop(torch.tanh(self.concat_cs(concat_text)))
            text = self.layer_norm(text) + self.bert_linear(merged_layer)
 
        elif self.opt.use_const_and_sememe == "pyramid2_layer_norm_FFN":
            const_tag_embed = self.const_embed(const)
            const_pos_embed = self.const_pos_embed(const_pos)
            const_info_embed = torch.sum(const_tag_embed * const_pos_embed, dim=2)

            sememe_embed = self.sememe_embed(sememe)
            sememe_embed = torch.mean(sememe_embed, dim=2)
            bsc_text = torch.sigmoid(self.bsc(torch.cat([merged_layer, const_info_embed, sememe_embed], dim=-1)))
            bs_text = torch.sigmoid(self.bs(torch.cat([merged_layer, sememe_embed], dim=-1)))
            bc_text = torch.sigmoid(self.bc(torch.cat([merged_layer, const_info_embed], dim=-1)))
            sc_text = torch.sigmoid(self.sc(torch.cat([sememe_embed, const_info_embed], dim=-1)))

            cs_text_0 = torch.concat([bsc_text, bs_text, bc_text, sc_text], dim=2)
            cs_text_1 = torch.tanh(self.p1(cs_text_0))
            cs_text_2 = torch.tanh(self.p2(cs_text_1))
            cs_text_3 = torch.tanh(self.p3(cs_text_2))
            concat_text = torch.concat([cs_text_0 * self.w0, cs_text_1 * self.w1, cs_text_2 * self.w2, cs_text_3 * self.w3], dim=2)
            text = torch.tanh(self.concat_cs(concat_text))
            text_FFN = self.fc2(self.drop(F.relu(self.fc1(text))))
            text = self.layer_norm(text + text_FFN) + self.bert_linear(merged_layer)
        
        if self.opt.use_const_sememe_dep == "pyramid2_layer_norm_drop":
            const_tag_embed = self.const_embed(const)
            const_pos_embed = self.const_pos_embed(const_pos)
            const_info_embed = torch.sum(const_tag_embed * const_pos_embed, dim=2)
            const_info_embed = self.cs_drop(const_info_embed)

            sememe_embed = self.sememe_embed(sememe)
            sememe_embed = torch.mean(sememe_embed, dim=2)
            sememe_embed = self.cs_drop(sememe_embed)


            dep_tag_embed = self.dep_embed(dep)
            dep_pos_embed = self.dep_pos_embed(dep_pos)
            dep_info_embed = torch.sum(dep_tag_embed * dep_pos_embed, dim=2)
            dep_info_embed = self.cs_drop(dep_info_embed)

            bscd_text = self.linear_drop(torch.sigmoid(self.bscd(torch.cat([merged_layer, const_info_embed, sememe_embed, dep_info_embed], dim=-1))))
            bs_text = self.linear_drop(torch.sigmoid(self.bs(torch.cat([merged_layer, sememe_embed], dim=-1))))
            bc_text = self.linear_drop(torch.sigmoid(self.bc(torch.cat([merged_layer, const_info_embed], dim=-1))))
            bd_text = self.linear_drop(torch.sigmoid(self.bd(torch.cat([merged_layer, dep_info_embed], dim=-1))))
            scd_text = self.linear_drop(torch.sigmoid(self.scd(torch.cat([sememe_embed, const_info_embed, dep_info_embed], dim=-1))))
            

            cs_text_0 = torch.concat([bscd_text, bs_text, bc_text, bd_text, scd_text], dim=2)
            cs_text_1 = self.linear_drop(torch.tanh(self.p1(cs_text_0)))
            cs_text_2 = self.linear_drop(torch.tanh(self.p2(cs_text_1)))
            cs_text_3 = self.linear_drop(torch.tanh(self.p3(cs_text_2)))
            concat_text = torch.concat([cs_text_0 * self.w0, cs_text_1 * self.w1, cs_text_2 * self.w2, cs_text_3 * self.w3], dim=2)
            text = self.linear_drop(torch.tanh(self.concat_cs(concat_text)))
            text = self.layer_norm(text) + self.bert_linear(merged_layer)

        if self.opt.use_const_and_dep == "pyramid2_layer_norm_drop":
            const_tag_embed = self.const_embed(const)
            const_pos_embed = self.const_pos_embed(const_pos)
            const_info_embed = torch.sum(const_tag_embed * const_pos_embed, dim=2)
            const_info_embed = self.cs_drop(const_info_embed)

            dep_tag_embed = self.dep_embed(dep)
            dep_pos_embed = self.dep_pos_embed(dep_pos)
            dep_info_embed = torch.sum(dep_tag_embed * dep_pos_embed, dim=2)
            dep_info_embed = self.cs_drop(dep_info_embed)

            bcd_text = self.linear_drop(torch.sigmoid(self.bcd(torch.cat([merged_layer, const_info_embed, dep_info_embed], dim=-1))))
            bc_text = self.linear_drop(torch.sigmoid(self.bc(torch.cat([merged_layer, const_info_embed], dim=-1))))
            bd_text = self.linear_drop(torch.sigmoid(self.bd(torch.cat([merged_layer, dep_info_embed], dim=-1))))
            cd_text = self.linear_drop(torch.sigmoid(self.cd(torch.cat([const_info_embed, dep_info_embed], dim=-1))))
            

            cs_text_0 = torch.concat([bcd_text, bc_text, bd_text, cd_text], dim=2)
            cs_text_1 = self.linear_drop(torch.tanh(self.p1(cs_text_0)))
            cs_text_2 = self.linear_drop(torch.tanh(self.p2(cs_text_1)))
            cs_text_3 = self.linear_drop(torch.tanh(self.p3(cs_text_2)))
            concat_text = torch.concat([cs_text_0 * self.w0, cs_text_1 * self.w1, cs_text_2 * self.w2, cs_text_3 * self.w3], dim=2)
            text = self.linear_drop(torch.tanh(self.concat_cs(concat_text)))
            text = self.layer_norm(text) + self.bert_linear(merged_layer)
        if self.opt.use_sememe_and_dep == "pyramid2_layer_norm_drop":

            sememe_embed = self.sememe_embed(sememe)
            sememe_embed = torch.mean(sememe_embed, dim=2)
            sememe_embed = self.cs_drop(sememe_embed)


            dep_tag_embed = self.dep_embed(dep)
            dep_pos_embed = self.dep_pos_embed(dep_pos)
            dep_info_embed = torch.sum(dep_tag_embed * dep_pos_embed, dim=2)
            dep_info_embed = self.cs_drop(dep_info_embed)

            bsd_text = self.linear_drop(torch.sigmoid(self.bsd(torch.cat([merged_layer, sememe_embed, dep_info_embed], dim=-1))))
            bs_text = self.linear_drop(torch.sigmoid(self.bs(torch.cat([merged_layer, sememe_embed], dim=-1))))
            bd_text = self.linear_drop(torch.sigmoid(self.bd(torch.cat([merged_layer, dep_info_embed], dim=-1))))
            sd_text = self.linear_drop(torch.sigmoid(self.sd(torch.cat([sememe_embed, dep_info_embed], dim=-1))))
            

            cs_text_0 = torch.concat([bsd_text, bs_text, bd_text, sd_text], dim=2)
            cs_text_1 = self.linear_drop(torch.tanh(self.p1(cs_text_0)))
            cs_text_2 = self.linear_drop(torch.tanh(self.p2(cs_text_1)))
            cs_text_3 = self.linear_drop(torch.tanh(self.p3(cs_text_2)))
            concat_text = torch.concat([cs_text_0 * self.w0, cs_text_1 * self.w1, cs_text_2 * self.w2, cs_text_3 * self.w3], dim=2)
            text = self.linear_drop(torch.tanh(self.concat_cs(concat_text)))
            text = self.layer_norm(text) + self.bert_linear(merged_layer)

        if self.opt.use_const_sememe_dep == "pyramid2_LN_D":
            const_tag_embed = self.const_embed(const)
            const_pos_embed = self.const_pos_embed(const_pos)
            const_info_embed = torch.sum(const_tag_embed * const_pos_embed, dim=2)
            const_info_embed = self.cs_drop(const_info_embed)

            sememe_embed = self.sememe_embed(sememe)
            sememe_embed = torch.mean(sememe_embed, dim=2)
            sememe_embed = self.cs_drop(sememe_embed)


            dep_tag_embed = self.dep_embed(dep)
            dep_pos_embed = self.dep_pos_embed(dep_pos)
            dep_info_embed = torch.sum(dep_tag_embed * dep_pos_embed, dim=2)
            dep_info_embed = self.cs_drop(dep_info_embed)

            bscd_text = self.linear_drop(torch.sigmoid(self.bscd(torch.cat([merged_layer, const_info_embed, sememe_embed, dep_info_embed], dim=-1))))
            bs_text = self.linear_drop(torch.sigmoid(self.bs(torch.cat([merged_layer, sememe_embed], dim=-1))))
            bc_text = self.linear_drop(torch.sigmoid(self.bc(torch.cat([merged_layer, const_info_embed], dim=-1))))
            bd_text = self.linear_drop(torch.sigmoid(self.bd(torch.cat([merged_layer, dep_info_embed], dim=-1))))

            

            cs_text_0 = torch.concat([bscd_text, bs_text, bc_text, bd_text], dim=2)
            cs_text_1 = self.linear_drop(torch.tanh(self.p1(cs_text_0)))
            cs_text_2 = self.linear_drop(torch.tanh(self.p2(cs_text_1)))
            cs_text_3 = self.linear_drop(torch.tanh(self.p3(cs_text_2)))
            concat_text = torch.concat([cs_text_0 * self.w0, cs_text_1 * self.w1, cs_text_2 * self.w2, cs_text_3 * self.w3], dim=2)
            text = self.linear_drop(torch.tanh(self.concat_cs(concat_text)))
            text = self.layer_norm(text) + self.bert_linear(merged_layer)
        if self.opt.use_const_sememe_dep == "attention":
            const_tag_embed = self.const_embed(const)
            const_pos_embed = self.const_pos_embed(const_pos)
            const_info_embed = self.cs_drop(torch.sum(const_tag_embed*const_pos_embed, dim=2))

            Q = self.const_Q(merged_layer)
            K = self.const_K(const_info_embed)
            V = self.const_V(merged_layer)
            attn = torch.matmul(Q, K.transpose(-2, -1))
            p_attn = self.att_drop(F.softmax(attn, dim=-1))
            const_text = torch.matmul(p_attn, V)

            sememe_embed = self.sememe_embed(sememe)
            sememe_embed = self.cs_drop(torch.mean(sememe_embed, dim=2))

            Q = self.sememe_Q(merged_layer)
            K = self.sememe_K(sememe_embed)
            V = self.sememe_V(merged_layer)
            attn = torch.matmul(Q, K.transpose(-2, -1))
            p_attn = self.att_drop(F.softmax(attn, dim=-1))
            sememe_text = torch.matmul(p_attn, V)

            dep_tag_embed = self.dep_embed(dep)
            dep_pos_embed = self.dep_pos_embed(dep_pos)
            dep_info_embed = torch.sum(dep_tag_embed * dep_pos_embed, dim=2)
            dep_info_embed = self.cs_drop(dep_info_embed)

            Q = self.dep_Q(merged_layer)
            K = self.dep_K(dep_info_embed)
            V = self.dep_V(merged_layer)
            attn = torch.matmul(Q, K.transpose(-2, -1))
            p_attn = self.att_drop(F.softmax(attn, dim=-1))
            dep_text = torch.matmul(p_attn, V)

            text = self.bert_linear(merged_layer) + self.layer_norm(const_text+sememe_text+dep_text)

        if self.opt.use_const_sememe_dep == "gate":
            raw_text = self.bert_linear(merged_layer)

            const_tag_embed = self.const_embed(const)
            const_pos_embed = self.const_pos_embed(const_pos)
            const_info_embed = self.cs_drop(torch.sum(const_tag_embed * const_pos_embed, dim=2))
            const_g = self.gate_drop(self.gate_activation(self.const_G(const_info_embed))) * raw_text

            sememe_embed = self.sememe_embed(sememe)
            sememe_embed = self.cs_drop(torch.mean(sememe_embed, dim=2))
            sememe_g = self.gate_drop(self.gate_activation(self.sememe_G(sememe_embed))) * raw_text


            dep_tag_embed = self.dep_embed(dep)
            dep_pos_embed = self.dep_pos_embed(dep_pos)
            dep_info_embed = self.cs_drop(torch.sum(dep_tag_embed * dep_pos_embed, dim=2))
            dep_g = self.gate_drop(self.gate_activation(self.dep_G(dep_info_embed))) * raw_text

            text = raw_text + self.layer_norm(const_g+sememe_g+dep_g)

        if text == None:
            text = self.bert_linear(merged_layer)
        text_out = self.text_embed_dropout(text)        
        text_out_fixed = text_out
         
        rl_input = text_out_fixed
         
        _, aspect_vec_fixed, nonaspect_mask = self.mask(rl_input, aspect_double_idx)
        nonaspect_mask = nonaspect_mask * mask
        
        
        
        initial_rank_scores = F.softmax(torch.matmul(bert_out.unsqueeze(dim=1), rl_input.transpose(1, 2)).masked_fill(mask.unsqueeze(dim=1)==0, -1e9), dim=2).squeeze(dim=1)
         
        trees, samples = self.rl_tree_generator(rl_input, words, word_seq_lengths, aspect_vec_fixed, aspect_double_idx, temperature=temperature, initial_rank_scores=initial_rank_scores)
        
        if not self.training: 
            rl_adj = torch.zeros(batch_size, seq_len, seq_len, device = text_indices.device)
            
            syn_dist =  torch.zeros(batch_size, seq_len, device = text_indices.device).fill_(-seq_len)
            
            rank_logits = torch.zeros(batch_size, seq_len, device = text_indices.device).fill_(-1e9)
            sample_rank_scores = samples['rank_scores']
            
            for b in range(batch_size):
                if debugger and text_len[b] < 30 and aux_aspect_targets.size(0) > 0: 
                    print(" ".join(words[b]))
                    print(labels[b])
                    print(sample_rank_scores[b].softmax(dim=-1).cpu().numpy().tolist())
                left, right = aspect_double_idx[b][0].item(), aspect_double_idx[b][1].item()
                if debugger: 
                    print(left, right)
                if debugger and text_len[b] < 30 and aux_aspect_targets.size(0) > 0: 
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
                
                if debugger: 
                    for k in range(text_len[b]):
                        self.dist_file.write(str(syn_dist[b][k].item())+" ")
                    self.dist_file.write("\n")
                
                for pair in pairs:
                    i, j, w = pair 
                    
                    rl_adj[b][i][j] = w
                    
                    if debugger: 
                        if i<j:# and ( left <=i <=right  or  left<=j <=right): 
                            debugger.adj_pred_sum += 1 
                            if adj[b][i][j] == 1: 
                                debugger.adj_common_sum += 1 
                
                if debugger: 
                    for m in range(text_len[b]):
                        for n in range(m+1, text_len[b]):
                            if adj[b][m][n] == 1:# and ( left <=m <=right  or  left<=n <=right):
                                debugger.adj_gold_sum += 1 
                
                        
                #print(rl_adj[b])
            logits, alpha, _, _  = self.classifier(text_out, bert_out, adj, rl_adj, aspect_double_idx, text_len, aspect_len, syn_dist, rank_logits)  #Batch size: 16 * 3
            #print(f"[tlog] logits: {logits.size()}")
            
            
            if debugger: 
                #print(aspect_double_idx)
                #print("alpha: " + str(alpha[0][0].cpu().numpy()))
                debugger.alpha = alpha
                batch_size, _, = alpha.size()
                for i in range(batch_size):
                    #print("alpha: " + str(alpha[b][0].cpu().numpy()))
                    #print(aspect_double_idx)
                    b, e = aspect_double_idx[i].cpu().numpy().tolist()
                    #print(b, e)
                    attention_list = alpha[i].cpu().numpy().tolist()
                    debugger.update_list(b, e, attention_list)
            
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
            tree_distance_regularized_loss = batch_attention_loss.mean()
            
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
             
            sample_num = self.opt.sample_num 
            aspect_double_idx_expanded = aspect_double_idx.unsqueeze(dim=1).repeat(1, sample_num, 1).view(batch_size * sample_num, -1)
            text_out_expanded = text_out.unsqueeze(dim=1).repeat(1, sample_num, 1, 1).view(batch_size * sample_num, seq_len, -1)
            text_len_expanded = text_len.unsqueeze(dim=1).repeat(1, sample_num).view(batch_size * sample_num)
            
            aspect_len_expanded = aspect_len.unsqueeze(dim=1).repeat(1, sample_num).view(batch_size * sample_num)
            #nonaspect_mask_expanded = nonaspect_mask.unsqueeze(dim=1).repeat(1, sample_num, 1).view(batch_size * sample_num, -1)
            mask_expanded = mask.unsqueeze(dim=1).repeat(1, sample_num, 1).view(batch_size * sample_num, -1)
            
            adj_expanded = adj.unsqueeze(dim=1).repeat(1, sample_num, 1,1).view(batch_size*sample_num, seq_len, seq_len)
            bert_out_expanded = bert_out.unsqueeze(dim=1).repeat(1, sample_num, 1).view(batch_size * sample_num, -1)
            
            probs, sample_trees = samples['probs'], samples['trees']
            sample_rank_scores = samples['rank_scores']
            sample_normalized_entropy = sum(samples['normalized_entropy'])
            
            rl_adj = torch.zeros(batch_size * sample_num, seq_len, seq_len, device = text_indices.device)
            
            syn_dist =  torch.zeros(batch_size * sample_num, seq_len, device = text_indices.device).fill_(-seq_len)
            
            rank_logits = torch.zeros(batch_size * sample_num, seq_len, device = text_indices.device).fill_(-1e9)
             
            
            for b in range(len(sample_trees)):
                 
                left, right = aspect_double_idx_expanded[b][0].item(), aspect_double_idx_expanded[b][1].item()
                 
                debug = False   
                if debug: 
                    if self.count % 50 == 0:
                        print(" ".join(words[b//sample_num]))
                        print(" ".join(words[b//sample_num][left:right+1]))
                        print(sample_rank_scores[b])
                        print(sample_trees[b].print())
                
                pairs = []
                sample_trees[b].adj(pairs, sample_trees[b].index, 0)
                
                rank_logits[b][0:sample_rank_scores[b].size(0)] = sample_rank_scores[b]
                
                distances = {}
                sample_trees[b].syn_distance(0, distances)
                
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
            
            logits = reshaped_sample_logits[:,0,:]
            
            sample_label_pred = sample_logits.max(1)[1]
             
            sample_label_gt = labels.unsqueeze(1).expand(-1, sample_num).contiguous().view(-1)
           
            syn_dist = F.softmax(syn_dist*2, dim=-1)
            
            
            batch_attention_loss = (self.kl_div(input=(sample_alphas+1e-9).log(), target=syn_dist)).sum(dim=-1)
            tree_distance_regularized_loss = batch_attention_loss.mean()
           
            batch_distill_loss =  ((self.kl_div(input=(F.softmax(rank_logits * 10.0, dim=-1)+1e-9).log(), target=sample_alphas.detach()) * mask_expanded).sum(dim=-1))
            
            distill_loss = batch_distill_loss.mean()
            
          
            sample_i_pairs = reshaped_sample_features[:,0,:]
            sample_j_pairs = reshaped_sample_features[:,1,:]
            
           
            contrastive_loss = self.nt_xent_criterion(sample_i_pairs, sample_j_pairs) #+ self.nt_xent_criterion(sample_i_pairs, sample_k_pairs) + self.nt_xent_criterion(sample_j_pairs, sample_k_pairs)
            
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
                #cand_rewards = rl_rewards[i*sample_num: (i+1)*sample_num]
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
                 
                reshaped_sample_gcn_outputs = sample_gcn_outputs.view(batch_size, sample_num, seq_len, -1)
                
                aux_aspect_x = self.get_features_for_aux_aspect(reshaped_sample_gcn_outputs[:,0,:,:], aux_aspect_targets) # B' * D
                 
                aux_output = self.fc_aux(aux_aspect_x)
                 
                aux_loss = 0.01 * self.criterion(aux_output, aux_aspect_targets[:,-1]).mean() #adv loss 
                 
        ###########################
        
        if self.training:
            loss = self.criterion(logits, labels).mean()
            
            loss = loss + tree_distance_regularized_loss * self.opt.td_weight #0.1
            
            
            if self.policy_trainable: 
                
                loss = loss + rl_loss  + self.opt.ent_weight * sample_normalized_entropy + distill_loss * self.opt.att_weight
               
            if self.opt.use_aux_aspect and aux_aspect_targets.size(0) > 0:
                loss = loss + aux_loss 
            
            
            return logits, loss
        else:
            return logits, alpha, samples
