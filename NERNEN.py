#!python
# coding=utf-8
'''
:Description  : ner nen model
:Version      : 1.0
:Author       : LilNeo
:Date         : 2022-03-12 23:36:04
:LastEditors  : wy
:LastEditTime : 2022-03-15 23:20:02
:FilePath     : /nernen/nernen.py
:Copyright 2022 LilNeo, All Rights Reserved. 
'''

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
class BertSoftmaxForNer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertSoftmaxForNer, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # BaseModelOutputWithPoolingAndCrossAttentions(
        #     last_hidden_state=tensor([[[0.2517, -0.0140, -0.3235, ..., -0.3253, -0.1374, -0.2869],
        #                                [0.0378, -0.2594, -0.4355, ..., -0.0270, 0.5246, 0.3070],
        #                                ...,
        #                                [0.2291, -0.6564, -0.2516, ..., -0.1350, 0.5700, 0.1549]],
        #                               ...,
        #                               [[0.2006, -0.6508, 0.1353, ..., 0.1274, 0.1768, 0.1282],
        #                                [-0.0202, -0.5634, 0.3302, ..., 0.6453, -0.2915, -0.0880],
        #                                ...,
        #                                [0.3397, -0.4853, 0.1383, ..., 0.1240, 0.1156, 0.5034]]],
        #                               device='cuda:0', grad_fn= < NativeLayerNormBackward0 >), # torch.Size([24, 58, 768])
        #     pooler_output = tensor([[2.1065e-02, 1.0974e-01, 9.3199e-04, ..., 3.2423e-02, -4.6972e-02, 9.9985e-01],
        #                             [-1.3258e-01, -1.1980e-01, 4.6302e-01, ..., 5.3873e-01, 3.3172e-02, 9.9984e-01],
        #                             ...,
        #                             [-7.2824e-02, 3.2234e-02, 5.7599e-01, ..., 4.9335e-01, -5.5849e-02, 9.9941e-01]],
        #                             device='cuda:0', grad_fn= < TanhBackward0 >), # torch.Size([24, 768])
        #     hidden_states = None,
        #     past_key_values = None,
        #     attentions = None,
        #     cross_attentions = None
        # )
        sequence_output = outputs[0]  # torch.Size([24, 58, 768])
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output) # torch.Size([24, 58, num_labels]) # TODO

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        # outputs[2:]：()
        # outputs：一个元素的元组
        # (
        #     一个大小为torch.Size([24, 58, num_labels])的tensor
        # )
        if labels is not None:
            # labels: tensor([[2, 2, 2, ..., 0, 0, 0],
            #                 [2, 2, 2, ..., 0, 0, 0],
            #                 ...,
            #                 [2, 1, 4, ..., 0, 0, 0]], device='cuda:0')  # torch.Size([24, 58])
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            if attention_mask is not None:
                # attention_mask: tensor([[1, 1, 1, ..., 0, 0, 0],
                #                         [1, 1, 1, ..., 0, 0, 0],
                #                         ...,
                #                         [1, 1, 1, ..., 0, 0, 0]], device='cuda:0') # torch.Size([24, 58])
                active_loss = attention_mask.contiguous().view(-1) == 1
                # active_loss:
                # tensor([True, True, True, ..., False, False, False,
                #         True, True, True, ..., False, False, False,
                #         ......,
                #         True, True, True, ..., False, False, False]) # torch.Size([24*58=1392]) # 共796个True
                # 其中796为当前batch每个样本的有效长度总和，sum([23, 50, 55, 31, 39, 57, 32, 38, 30, 17, 50, 40, 29, 13, 4, 9, 31, 39, 40, 43, 54, 6, 58, 8]) = 796
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                # tensor([[-0.1009, 0.5519, -0.2233, 0.0435, -0.2085, 0.0846],
                #         [-0.0733, 0.4331, -0.1166, 0.1131, -0.2569, -0.0587],
                #         ...,
                #         [0.4100, -0.0364, 0.4357, -0.2085, 0.1305, -0.7824]],
                #        device='cuda:0', grad_fn= < IndexBackward0 >) # torch.Size([796, num_labels])
                active_labels = labels.contiguous().view(-1)[active_loss]
                # tensor([2, 2, 2, ..., 2, 2, 2, # 长23
                #         2, 2, 2, ..., 2, 2, 2, # 长50
                #         ...,
                #         2, 1, 4, ..., 2, 2, 2], # 长8
                #        device='cuda:0') # torch.Size([796])
                loss = loss_fct(active_logits, active_labels) # tensor(1.9131, device='cuda:0', grad_fn=<NllLossBackward0>)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # loss, scores, (hidden_states), (attentions)

class BertSoftmaxForNen(BertPreTrainedModel): # multitask
    def __init__(self, config):
        super(BertSoftmaxForNen, self).__init__(config)
        self.num_labels = config.num_labels # 主任务nen
        self.num_labels2 = config.num_labels2  # 辅助任务ner
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*2, config.num_labels) # nen主任务 # TODO
        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier2 = nn.Linear(config.hidden_size, config.num_labels2) # ner
        # self.fc = nn.Linear(config.hidden_size*2, config.hidden_size) # TODO
        # self.relu = nn.ReLU(inplace=True)
        self.label_embedding = nn.Embedding(config.num_labels2, config.hidden_size)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, ner_labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        # sequence_output = self.dropout(sequence_output)
        ner_embedding = self.label_embedding(ner_labels)*10
        ccat = torch.cat((sequence_output , ner_embedding),-1)
        ccat = self.dropout(ccat)
        # logits = self.fc(ccat)
        # logits = self.relu(logits)
        # logits = self.dropout(logits)
        # logits = self.classifier(logits)
        logits = self.classifier(ccat) # nen # TODO
        logits2 = self.classifier2(sequence_output) # ner
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None and ner_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            loss_fct2 = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss] # nen
                active_logits2 = logits2.view(-1, self.num_labels2)[active_loss] # ner
                active_labels = labels.view(-1)[active_loss]
                active_labels2 = ner_labels.view(-1)[active_loss]
                # loss = loss_fct(active_logits, active_labels) + loss_fct2(active_logits2, active_labels2)
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1)) + loss_fct2(logits2.view(-1, self.num_labels2), ner_labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)



class Transfernet(nn.Module):
    def __init__(self, config):
        super(Transfernet, self).__init__(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.relu = nn.ReLu()
        self.init_weights()
        self.loss = torch.nn.MSELoss()

    def forward(self, x, y):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        loss = self.loss(x, y)
        return loss, x