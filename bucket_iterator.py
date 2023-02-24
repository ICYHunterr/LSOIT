# -*- coding: utf-8 -*-

import math
import random
import torch
import numpy

class BucketIterator(object):
    def __init__(self, data, batch_size, sort_key='text_indices', shuffle=True, sort=True):
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.ma_dict = {}
        self.save_multiaspect(data)
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            #print("[tlog] data " + str(data[:10]))
            sorted_data = sorted(data, key=lambda x: len(x[self.sort_key])) #tzy: from small to big 
            #print("[tlog] sorted_data " + str(sorted_data[:10]))
            #sys.exit(0)
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batches
    
    def save_multiaspect(self, data):
        
        for example in data:
            #print(example)
            sent_id, span, polarity = example['sent_id'], example['span'], example['polarity']
            if not sent_id in self.ma_dict:
                self.ma_dict[sent_id] = set()
            aspect_tuple = (sent_id, span[0], span[1], polarity)
            if aspect_tuple not in self.ma_dict:
                self.ma_dict[sent_id].add(aspect_tuple)
        
        num_ma = 0 #number of multiple aspects 
        
        for sent_id in self.ma_dict:
            aspects = self.ma_dict[sent_id]
            if len(aspects) > 1:
                num_ma+=1
        print("[tlog] multiple aspects: " + str(num_ma))
        #sys.exit(0)
        
    def pad_data(self, batch_data):
        batch_text_indices = []
        batch_pos_indices = []
        batch_rel_indices = []
        batch_context_indices = []
        batch_aspect_indices = []
        batch_aspect_bert_indices = []
        batch_left_indices = []
        batch_left_bert_indices = []
        batch_polarity = []
        batch_dependency_graph = []
        batch_text_bert_indices =[] 
        batch_labeled_bert_indices =[] 
        batch_text_raw_bert_indices =[] 
        
        batch_bert_segments_ids =[] 
        batch_bert_token_masks = []
        
        batch_labeled_bert_segments_ids =[] 
        batch_labeled_bert_token_masks = []
        
        batch_word_lens = []

        batch_const_indices = []
        batch_const_pos_indices = []

        batch_sememe_indices = []

        batch_dep_indices = []
        batch_dep_pos_indices = []

        # 词的句法最大路径长度
        max_const_path_len = 0
        for indices in batch_data:
            max_const_path_len = max(max_const_path_len, max([len(path) for path in indices["const_indices"]]))

        # 词的义原最大个数
        max_sememe_len = 0
        for indices in batch_data:
            max_sememe_len = max(max_sememe_len, max([len(sem) for sem in indices["sememe_indices"]]))

        # 依存路径最大路径长度
        max_dep_path_len = 0
        for indices in batch_data:
            max_dep_path_len = max(max_dep_path_len, max([len(path) for path in indices["dep_indices"]]))

        max_len = max([len(t[self.sort_key]) for t in batch_data])
        max_bert_len = max([len(t['text_bert_indices']) for t in batch_data])
        max_labeled_bert_len = max([len(t['labeled_bert_indices']) for t in batch_data])
        max_raw_bert_len = max([len(t['text_raw_bert_indices']) for t in batch_data])
        batch_aux_aspect_targets = []
        batch_words = []
        batch_dist_to_target = []
        batch_text = []
        
        
        #print("[tlog] max_len, max_bert_len, max_raw_bert_len " + str(max_len) + ", " + str(max_bert_len) + ", " + str(max_raw_bert_len) )
        batch_index = 0
        for item in batch_data:
            text_indices, context_indices, aspect_indices, aspect_bert_indices, left_indices, left_bert_indices, \
            polarity, dependency_graph, pos_indices, rel_indices, text_bert_indices, labeled_bert_indices, \
            text_raw_bert_indices, bert_segments_ids, bert_token_masks, labeled_bert_segments_ids, \
            labeled_bert_token_masks, word_lens, words, dist_to_target = \
                item['text_indices'], item['context_indices'], item['aspect_indices'], item['aspect_bert_indices'], \
                item['left_indices'], item['left_bert_indices'], item['polarity'], item['dependency_graph'], \
                item['pos_indices'], item['rel_indices'], item['text_bert_indices'], item['labeled_bert_indices'], \
                item['text_raw_bert_indices'], item['bert_segments_ids'], item['bert_token_masks'], \
                item['labeled_bert_segments_ids'], item['labeled_bert_token_masks'], item['word_lens'], item['words'], \
                item['dist_to_target']
            const_indices, const_pos_indices = item["const_indices"], item["const_pos_indices"]
            sememe_indices = item["sememe_indices"]
            dep_indices, dep_pos_indices = item["dep_indices"], item["dep_pos_indices"]

            text = item["text"]
            batch_text.append(text)

            text_padding = [0] * (max_len - len(text_indices))
            rel_padding = [0] * (max_len - len(rel_indices))
            pos_padding = [0] * (max_len - len(pos_indices))
            
            context_padding = [0] * (max_len - len(context_indices))
            aspect_padding = [0] * (max_len - len(aspect_indices))
            left_padding = [0] * (max_len - len(left_indices))
            
            left_bert_padding = [0] * (max_bert_len -len(left_bert_indices))
            aspect_bert_padding = [0] * (max_bert_len - len(aspect_bert_indices))

            const_filled = []
            for const_indice in const_indices:
                const_padding = [0] * (max_const_path_len - len(const_indice))
                const_filled.append(const_indice + const_padding)
            
            const_pos_filled = []
            for const_pos_indice in const_pos_indices:
                const_padding = [0] * (max_const_path_len - len(const_pos_indice))
                const_pos_filled.append(const_pos_indice + const_padding)


            sememe_filled = []
            for sememe_indice in sememe_indices:
                sememe_padding = [0] * (max_sememe_len - len(sememe_indice))
                sememe_filled.append(sememe_indice + sememe_padding)

            dep_filled = []
            for dep_indice in dep_indices:
                dep_padding = [0] * (max_dep_path_len - len(dep_indice))
                dep_filled.append(dep_indice + dep_padding)
            
            dep_pos_filled = []
            for dep_pos_indice in dep_pos_indices:
                dep_padding = [0] * (max_dep_path_len - len(dep_pos_indice))
                dep_pos_filled.append(dep_pos_indice + dep_padding)

            for n in range(len(text_padding)):
                const_filled.append([0]*max_const_path_len)
                const_pos_filled.append([0]*max_const_path_len)
                sememe_filled.append([0]*max_sememe_len)
                dep_filled.append([0]*max_dep_path_len)
                dep_pos_filled.append([0]*max_dep_path_len)
            

            batch_const_pos_indices.append(const_pos_filled)
            batch_const_indices.append(const_filled)
            batch_sememe_indices.append(sememe_filled)
            batch_dep_indices.append(dep_filled)
            batch_dep_pos_indices.append(dep_pos_filled)
            

            batch_text_indices.append(text_indices + text_padding)
            batch_pos_indices.append(pos_indices + pos_padding)
            batch_rel_indices.append(rel_indices + rel_padding)
            
            batch_context_indices.append(context_indices + context_padding)
            batch_aspect_indices.append(aspect_indices + aspect_padding)
            batch_aspect_bert_indices.append(aspect_bert_indices + aspect_bert_padding)
            batch_left_indices.append(left_indices + left_padding)
            batch_left_bert_indices.append(left_bert_indices + left_bert_padding)
            #print("[tlog] text_bert_indices: " + str(text_bert_indices))
            #print("[tlog] text_raw_bert_idices: " + str(text_raw_bert_indices))
            #print("[tlog] bert_segments_ids: " + str(bert_segments_ids))
            
            batch_text_bert_indices.append(text_bert_indices + [0] * (max_bert_len - len(text_bert_indices)))
            batch_labeled_bert_indices.append(labeled_bert_indices + [0] * (max_labeled_bert_len - len(labeled_bert_indices)))
            
            batch_text_raw_bert_indices.append(text_raw_bert_indices + [0] * (max_raw_bert_len - len(text_raw_bert_indices)))
            
            batch_bert_segments_ids.append(bert_segments_ids + [1] * (max_bert_len - len(bert_segments_ids)))
            batch_bert_token_masks.append(bert_token_masks + [0] * (max_bert_len - len(bert_token_masks)))
            
            batch_labeled_bert_segments_ids.append(labeled_bert_segments_ids + [1] * (max_labeled_bert_len - len(labeled_bert_segments_ids)))
            batch_labeled_bert_token_masks.append(labeled_bert_token_masks + [0] * (max_labeled_bert_len - len(labeled_bert_token_masks)))
            
            batch_word_lens.append(word_lens)
            batch_words.append(words)
            #batch_dist_to_target.append(dist_to_target + [max_len] * (max_len  - len(dist_to_target)))
            batch_dist_to_target.append(dist_to_target + [0] * (max_len  - len(dist_to_target)))
            
            batch_polarity.append(polarity)
            batch_dependency_graph.append(numpy.pad(dependency_graph, \
                ((0, max_len-len(text_indices)),(0, max_len-len(text_indices))), 'constant'))
            
            sent_id = item['sent_id']
            aspect_span = item['span']
            assert aspect_span[0] < max_len and aspect_span[1] < max_len 
            ma_set = self.ma_dict[sent_id]
            if len(ma_set)> 1:
                for aspect_example in ma_set:
                    sid, span_start, span_end, polarity = aspect_example
                    assert sid == sent_id 
                    assert span_start < max_len and span_end < max_len
                    if span_start != aspect_span[0] and span_end != aspect_span[1]: 
                        batch_aux_aspect_targets.append([batch_index, span_start, span_end, polarity])
            batch_index += 1
        return { \
                'text_indices': torch.tensor(batch_text_indices), \
                'context_indices': torch.tensor(batch_context_indices), \
                'aspect_indices': torch.tensor(batch_aspect_indices), \
                'aspect_bert_indices': torch.tensor(batch_aspect_bert_indices), \
                'left_indices': torch.tensor(batch_left_indices), \
                'left_bert_indices': torch.tensor(batch_left_bert_indices), \
                'polarity': torch.tensor(batch_polarity), \
                'dependency_graph': torch.tensor(batch_dependency_graph),\
                'pos_indices': torch.tensor(batch_pos_indices),\
                'rel_indices': torch.tensor(batch_rel_indices),\
                'text_bert_indices': torch.tensor(batch_text_bert_indices), \
                'labeled_bert_indices': torch.tensor(batch_labeled_bert_indices), \
                'text_raw_bert_indices': torch.tensor(batch_text_raw_bert_indices), \
                'bert_segments_ids': torch.tensor(batch_bert_segments_ids), \
                'bert_token_masks': torch.tensor(batch_bert_token_masks),\
                'labeled_bert_segments_ids': torch.tensor(batch_labeled_bert_segments_ids), \
                'labeled_bert_token_masks': torch.tensor(batch_labeled_bert_token_masks),\
                'word_lens': batch_word_lens,\
                'words': batch_words,\
                'aux_aspect_targets': torch.tensor(batch_aux_aspect_targets),\
                'dist_to_target': torch.tensor(batch_dist_to_target),\
                "const_indices":torch.tensor(batch_const_indices),\
                "const_pos_indices":torch.tensor(batch_const_pos_indices),\
                "sememe_indices":torch.tensor(batch_sememe_indices),\
                "text":batch_text,\
                "dep_indices":torch.tensor(batch_dep_indices),\
                'dep_pos_indices':torch.tensor(batch_dep_pos_indices),\
            }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]
