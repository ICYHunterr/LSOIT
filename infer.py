# -*- coding: utf-8 -*-

import sys
import math
from sklearn import metrics

import torch
import torch.nn.functional as F

from data_utils import ABSADatesetReader, ABSADataset, build_embedding_matrix
from bucket_iterator import BucketIterator
from dependency_graph_stanza_towe import dependency_adj_matrix
from collections import defaultdict
from models import BertSPC, RLGCN, RLVAEGCN, DepGCNv2

class AttentionDebuger:
    def __init__(self):
        self.dist_word_dict = defaultdict(int)
        self.dist_attention_weights = defaultdict(float)
        self.dice_loss1 = 0
        self.dice_loss2 = 0
        self.kuma_adj = None
        self.alpha = None
        self.adj_gold_sum = 0.0
        self.adj_pred_sum = 0.0
        self.adj_common_sum = 0.0
    
    def clear(self):
        self.kuma_adj = None
        self.alpha = None
    
    def update(self, i, b, e, s):
        dist = 0 
        s = max(s, 1e-25)
        if math.isnan(s):
            s = 0 
        if i < b: 
            dist = b - i 
        elif i > e: 
            dist = i - e 
        else: 
            dist = 0
        dist = self.bin_att_len(dist)
        self.dist_word_dict[dist] += 1 
        self.dist_attention_weights[dist] += s 
        
    def update_list(self, b, e, s_list):
        for i in range(len(s_list)):
            self.update(i, b, e, s_list[i])
    
    def bin_att_len(self, sent_len):
        if sent_len <= 7:
            return sent_len 
        else:
            return 7
        return 16
    
    def report_uas_score(self):
        p = self.adj_common_sum*1.0/self.adj_pred_sum 
        r = self.adj_common_sum*1.0/self.adj_gold_sum 
        f = 2*p*r/(p+r)
        print(f"P/R/F： {p} \t {r} \t {f}")
    def report(self):
        keys = self.dist_word_dict.keys()
        keys = sorted(keys)
        for dist in keys:
            average_score = self.dist_attention_weights[dist] / self.dist_word_dict[dist]
            print("{}\t{}\t{}".format(dist, self.dist_word_dict[dist], average_score))
        
        self.report_uas_score()
        
class Inferer:
    """A simple inference example"""
    def __init__(self, opt):
        self.opt = opt
        absa_dataset = ABSADatesetReader(dataset=opt.dataset, embed_dim=opt.embed_dim)
        self.absa_dataset = absa_dataset
        self.train_data_loader = BucketIterator(data=absa_dataset.train_data, batch_size=opt.batch_size, shuffle=True)
        self.test_data_loader = BucketIterator(data=absa_dataset.test_data, batch_size=opt.batch_size, shuffle=False, sort=False)
        self.dev_data_loader = BucketIterator(data=absa_dataset.dev_data, batch_size=opt.batch_size, shuffle=False, sort=False)
        
        opt.pos_size = len(absa_dataset.pos_tokenizer.word2idx)
        opt.rel_size = len(absa_dataset.rel_tokenizer.word2idx)
        opt.pos_dim = 30
        opt.rel_dim = 30
        opt.const_size = len(absa_dataset.const_tokenizer.word2idx)
        opt.const_pos_size = len(absa_dataset.const_pos_tokenizer.word2idx)
        self.model = opt.model_class(absa_dataset.embedding_matrix, opt).to(opt.device)
    
        if torch.cuda.is_available():
            print('cuda memory allocated:', torch.cuda.memory_allocated(device=opt.device.index))
            
        print('loading model {0} ...'.format(opt.model_name))
        use_single = True    
        if use_single: 
            self.model.load_state_dict(torch.load(opt.state_dict_path))
        else:
            model_prefix = opt.state_dict_path.replace(".pkl", "")
            print("[tlog] model_prefix: " + str(model_prefix))
            from models.average import average_checkpoints
            model_name2 = [ model_prefix + "." + str(x) + ".pkl"  for x in range(0,1)]
            print(model_name2)
            print("[tlog] average")
            model_state = average_checkpoints(model_name2)
            self.model.load_state_dict(model_state)

        print("[tlog] load success")
        self.model.eval()
        torch.autograd.set_grad_enabled(False)
    
    def evaluate_acc_f1(self):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        debugger = AttentionDebuger()
        
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                t_inputs = [t_sample_batched[col].to(opt.device) if (col !="word_lens" and col != "words") else t_sample_batched[col] for col in self.opt.inputs_cols] 
                
                t_targets = t_sample_batched['polarity'].to(opt.device)
                
                t_outputs = self.model(t_inputs, labels=t_targets, debugger=debugger)

                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_test_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        debugger.report()
        return test_acc, f1
    
    def evaluate_acc_f1v2(self):
         # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        debugger = AttentionDebuger()
        
        test_data = self.absa_dataset.test_data
        len_acc_stat = defaultdict(int)
        len_stat = defaultdict(int)
        
        aspect_acc_stat = defaultdict(int)
        aspect_stat = defaultdict(int)
        
        debugger = AttentionDebuger()
        start_index = 0
        
        kuma_adj_dict = {}
        alpha_dict = {}
        
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                t_inputs = [t_sample_batched[col].to(opt.device) if (col !="word_lens" and col != "words") else t_sample_batched[col] for col in self.opt.inputs_cols] 
                t_targets = t_sample_batched['polarity'].to(opt.device)
                t_outputs = self.model(t_inputs, debugger=debugger, labels=t_targets)

                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                assert len(t_outputs) == len(t_targets)
                
                n_test_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)
                
                for j in range(t_targets.size(0)):
                    index = start_index + j 
                    sent_id = test_data[index]['sent_id']
                    sent_id = int(sent_id)
                    print("sent_id: " + str(test_data[index]['sent_id']))
                    aspects = self.test_data_loader.ma_dict[sent_id]
                    num_aspects = len(aspects)
                    print("[tlog] num aspects: " + str(num_aspects))

                    if num_aspects > 7:
                        num_aspects = 7 
                        
                    dependency_graph = torch.tensor([test_data[index]['dependency_graph']])

                    if(len(test_data[index]['text'].split()) != dependency_graph[0].size(0)):
                        print("[tlog] error")
                        print(len(test_data[index]['text']))
                        print(dependency_graph[0].size())

                    print("text:\n" + str(test_data[index]['text']))

                    if hasattr(debugger, 'alpha'):
                        alpha = debugger.alpha
                        if alpha is not None: 
                            attention = alpha[j].cpu().numpy().tolist()
                            attention = [ str(x) for x in attention]
                            print("alpha: " + " ".join(attention))

                    pred_label_index = t_outputs[j].argmax(axis=-1)
                    ploarity_label_index = test_data[index]['polarity']
                    print("predict: " + str(pred_label_index.item()))
                    print("gold   : " + str(ploarity_label_index))
            
                    sent_len = len(test_data[index]['text'].split())
                    sent_len = self.bin_len(sent_len)
            
                    aspect_stat[num_aspects] +=1 
                    len_stat[sent_len] += 1 
                    if pred_label_index == ploarity_label_index:
                        print("correct")
                        len_acc_stat[sent_len] += 1
                        aspect_acc_stat[num_aspects] += 1
                    else:
                        print("wrong")

                start_index += len(t_targets)
                debugger.clear()
                
        print(n_test_total) 
        print("[tlog] len acc stat: ")
        print(len_stat)
        print(len_acc_stat)
        
        for len_key in sorted(len_stat): 
            len_acc = len_acc_stat[len_key] * 1.0 / len_stat[len_key]
            print("%d\t%d\t%d\t%.2f" %(len_key, len_acc_stat[len_key], len_stat[len_key], len_acc))
            
        print("[tlog] aspect acc stat: ")
        for num_aspect_key in sorted(aspect_stat): 
            aspect_acc = aspect_acc_stat[num_aspect_key] * 1.0 / aspect_stat[num_aspect_key]
            print("%d\t%d\t%d\t%.2f" %(num_aspect_key, aspect_acc_stat[num_aspect_key], aspect_stat[num_aspect_key], aspect_acc))
        #print(sum([x[1] for x in len_acc_stat]))
        
        print("[tlog] final_acc: ")
        all_acc = sum([len_acc_stat[x] for x in len_acc_stat]) * 1.0 / sum([len_stat[x] for x in len_stat])
        print(all_acc)
        print("[tlog] attention weights")
        debugger.report()

        test_acc = n_test_correct / n_test_total
        test_f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        print('test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))
        return test_acc, test_f1
    
    def evaluate_test(self):
        test_acc, test_f1 = self.evaluate_acc_f1v2()  
        print('test_acc: {:.4f}, test_f1: {:.4f}'.format(test_acc, test_f1))
        sys.exit(0)
        test_data = self.absa_dataset.test_data
        #print(test_data[0])
         
        #print(raw_text)
        #text_seqs = [self.tokenizer.text_to_sequence(raw_text.lower())]
        #aspect_seqs = [self.tokenizer.text_to_sequence(aspect.lower())]
        #left_seqs = [self.tokenizer.text_to_sequence(raw_text.lower().split(aspect.lower())[0])]
        
        #print(text_seqs)
        #print(aspect_seqs)
        #print(left_seqs)
        #sys.exit(0)
        len_acc_stat = defaultdict(int)
        len_stat = defaultdict(int)
        
        aspect_acc_stat = defaultdict(int)
        aspect_stat = defaultdict(int)
        
        debugger = AttentionDebuger()
        
        for index in range(len(test_data)):
            text_indices = torch.tensor([test_data[index]['text_indices']], dtype=torch.int64)
            context_indices = torch.tensor([test_data[index]['context_indices']], dtype=torch.int64)
            aspect_indices = torch.tensor([test_data[index]['aspect_indices']], dtype=torch.int64)
            aspect_bert_indices = torch.tensor([test_data[index]['aspect_bert_indices']], dtype=torch.int64)
            left_indices = torch.tensor([test_data[index]['left_indices']], dtype=torch.int64)
            left_bert_indices = torch.tensor([test_data[index]['left_bert_indices']], dtype=torch.int64)
            #dependency_graph = torch.tensor([dependency_adj_matrix(raw_text.lower())])
            dependency_graph = torch.tensor([test_data[index]['dependency_graph']])
            pos_indices = torch.tensor([test_data[index]['pos_indices']], dtype=torch.int64)
            rel_indices = torch.tensor([test_data[index]['rel_indices']], dtype=torch.int64)
            text_bert_indices = torch.tensor([test_data[index]['text_bert_indices']], dtype=torch.int64)
            text_raw_bert_indices = torch.tensor([test_data[index]['text_raw_bert_indices']], dtype=torch.int64)
            bert_segments_ids = torch.tensor([test_data[index]['bert_segments_ids']], dtype=torch.int64)
            bert_token_masks = torch.tensor([test_data[index]['bert_token_masks']], dtype=torch.int64)
            word_lens = torch.tensor([test_data[index]['word_lens']], dtype=torch.int64)
            
            print("sent_id: " + str(test_data[index]['sent_id']))
            sent_id = test_data[index]['sent_id']
            sent_id = int(sent_id)
            aspects = self.test_data_loader.ma_dict[sent_id]
            num_aspects = len(aspects)
            print("[tlog] num aspects: " + str(num_aspects))
            #sys.exit(0)
            if num_aspects > 7:
                num_aspects = 7 
                
            assert num_aspects <=7
            
            print("text:\n" + str(test_data[index]['text']))
            #print("dep_graph:\n" + str(dependency_graph[0].numpy()))
            #sys.exit(0)
            data = {
                'text_indices': text_indices, 
                'context_indices': context_indices,
                'aspect_indices': aspect_indices,
                'aspect_bert_indices': aspect_bert_indices,
                'left_indices': left_indices, 
                'left_bert_indices': left_bert_indices, 
                'dependency_graph': dependency_graph,
                'pos_indices': pos_indices, 
                'rel_indices': rel_indices,
                'text_bert_indices': text_bert_indices, 
                'text_raw_bert_indices': text_raw_bert_indices,
                'bert_segments_ids': bert_segments_ids,
                'bert_token_masks': bert_token_masks,
                'word_lens': word_lens,
                'aux_aspect_targets': torch.tensor([], dtype=torch.int64)
            }
            #print(data)
            
            t_inputs = [data[col].to(opt.device) if col !="word_lens" else data[col] for col in self.opt.inputs_cols]
            t_outputs = self.model(t_inputs, debugger)

            t_probs = F.softmax(t_outputs, dim=-1).cpu().numpy()
            pred_label_index = t_probs.argmax(axis=-1)[0]
            ploarity_label_index = test_data[index]['polarity']
            print("predict: " + str(pred_label_index.item()))
            print("gold   : " + str(ploarity_label_index))
            
            sent_len = len(test_data[index]['text'].split())
            sent_len = self.bin_len(sent_len)
            
            aspect_stat[num_aspects] +=1 
            len_stat[sent_len] += 1 
            if pred_label_index == ploarity_label_index:
                print("correct")
                len_acc_stat[sent_len] += 1
                aspect_acc_stat[num_aspects] += 1
            else:
                print("wrong")
                
        print("[tlog] len acc stat: ")
        print(len_stat)
        print(len_acc_stat)
        
        for len_key in sorted(len_stat): 
            len_acc = len_acc_stat[len_key] * 1.0 / len_stat[len_key]
            print("%d\t%d\t%d\t%.2f" %(len_key, len_acc_stat[len_key], len_stat[len_key], len_acc))
            
        print("[tlog] aspect acc stat: ")
        for num_aspect_key in sorted(aspect_stat): 
            aspect_acc = aspect_acc_stat[num_aspect_key] * 1.0 / aspect_stat[num_aspect_key]
            print("%d\t%d\t%d\t%.2f" %(num_aspect_key, aspect_acc_stat[num_aspect_key], aspect_stat[num_aspect_key], aspect_acc))
        #print(sum([x[1] for x in len_acc_stat]))
        
        print("[tlog] final_acc: ")
        all_acc = sum([len_acc_stat[x] for x in len_acc_stat]) * 1.0 / sum([len_stat[x] for x in len_stat])
        print(all_acc)
        print("[tlog] attention weights")
        debugger.report()
        return t_probs

    def bin_len(self, sent_len):
        if sent_len >0 and sent_len <=10: 
            return 0
        if sent_len >10 and sent_len <=20: 
            return 1
        if sent_len >20 and sent_len <=30: 
            return 2
        if sent_len >30 and sent_len <=40: 
            return 3
        if sent_len >40 and sent_len <=50: 
            return 4
        if sent_len >50 and sent_len <=60: 
            return 5
        return 6
    
if __name__ == '__main__':
    dataset = sys.argv[1]
    print("[tlog] dataset: " + dataset)
    if len(sys.argv) == 4: 
        in_dataset = sys.argv[3]
    else:
        in_dataset = dataset
    # set your trained models here
    model_state_dict_paths = {
        'bert-spc': 'saved_models/state_dict_bert-spc/bert-spc_'+in_dataset+'.0.pkl',
        'rlgcn': 'saved_models/state_dict_rlgcn/rlgcn_'+in_dataset+'.0.pkl',
        'depgcn2': 'saved_models/state_dict_depgcn2/depgcn2_'+in_dataset+'.0.pkl',
    }
    model_classes = {
        'bert-spc': BertSPC,
        'depgcn2': DepGCNv2,
        'rlgcn': RLGCN
    }
    input_colses = {
        'bert-spc': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph', 'pos_indices', 'rel_indices', 'text_bert_indices', 'text_raw_bert_indices', 'bert_segments_ids', 'bert_token_masks', 'word_lens'],
        'rlgcn': ['text_indices', 'aspect_indices', 'aspect_bert_indices', 'left_indices', 'left_bert_indices', 'dependency_graph', 'pos_indices', 'rel_indices', 'text_bert_indices', 'text_raw_bert_indices', 'bert_segments_ids', 'bert_token_masks', 'word_lens', 'words', 'aux_aspect_targets'],
        'depgcn2': ['text_indices', 'aspect_indices', 'aspect_bert_indices', 'left_indices', 'left_bert_indices', 'dependency_graph', 'pos_indices', 'rel_indices', 'text_bert_indices', 'text_raw_bert_indices', 'bert_segments_ids', 'bert_token_masks', 'word_lens', 'words', 'aux_aspect_targets', 'dist_to_target']
    }
    class Option(object): pass
    opt = Option()
    opt.model_name = sys.argv[2] #'bert-spc'#'rlgcn'#'bert-spc'#'rlgcn' #'bert-spc'#
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.dataset = dataset
    opt.state_dict_path = model_state_dict_paths[opt.model_name]
    print("[tlog] opt.state_dict_path: " + str(opt.state_dict_path ))
    opt.use_single_bert = False  
    opt.embed_dim = 300
    opt.hidden_dim = 300
    opt.polarities_dim = 3
    opt.initializer = 'xavier_uniform_'
    opt.use_aux_aspect = False 
    #opt.head = 
    opt.lambda_p = 0.8
    opt.batch_size = 16
    opt.sample_num = 3 
    opt.use_bert_adam = True 
    opt.use_single_optimizer = False 
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inf = Inferer(opt)
    #sys.exit(0)
    #t_probs = inf.evaluate('The staff should be a bit more friendly .', 'staff')
    inf.evaluate_test()
    #print(t_probs)
    #print(t_probs.argmax(axis=-1)[0])

