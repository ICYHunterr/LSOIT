# -*- coding: utf-8 -*-

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel

class BertSPC(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(BertSPC, self).__init__()
        self.opt = opt
        # model_name = "indobenchmark/indobert-base-p2"
        model_name = "bert-base-uncased"
        self.bert_model = BertModel.from_pretrained(model_name, output_hidden_states=True, return_dict=False)
        self.bert_dim = 768
        self.fc = nn.Linear(self.bert_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.1)
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, labels = None,  debugger=None, temperature=None):
        text_indices, aspect_indices, left_indices, adj, pos_indices, rel_indices, text_bert_indices, text_raw_bert_indices, bert_segments_ids, bert_token_masks, word_seq_lengths = inputs

        _, pooled_output, encoded_layers = self.bert_model(input_ids=text_bert_indices, token_type_ids=bert_segments_ids, attention_mask=bert_token_masks)
        #text_out = self.bert_linear(pooled_output)
        text_out = pooled_output
        text_out = self.text_embed_dropout(text_out)
        logits = self.fc(text_out)
        if self.training: 
            #loss, optional = self.get_selection_loss(text_out, adj)
            loss = self.criterion(logits, labels).mean() 
            return logits, loss 
        else: 
            return logits
