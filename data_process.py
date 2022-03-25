# import codecs
# from operator import getitem
# import numpy as np
from tqdm import trange
# from typing import List, Dict
# from conlleval import evaluate
# from entitybase.entity_base_loader import EntityBase
# from keras.preprocessing.sequence import pad_sequences
# from sklearn.metrics import precision_score, recall_score, f1_score

import torch
from typing import List, Dict
import copy
import pickle

BASEPATH = "./datasets/"
class CDRProcessor(object):
    def __init__(self, tokenizer, dict_dataset: Dict, max_len):
        self.tokenizer = tokenizer
        self.__dict_dataset = dict_dataset # {'train': 'CDR/train.txt', 'dev': 'CDR/dev.txt', 'test': 'CDR/test.txt', 'zs_test': 'CDR/zs_test.txt'}
        # self.__entity_base = EntityBase(bert_path)
        # self.__dict_ner_label = ['X']
        self.__dict_ner_label = ['X', 'B-Chemical', 'O', 'B-Disease', 'I-Chemical', 'I-Disease']
        # 对训练集__parse_data后self.__dict_ner_label: ['X', 'B-Chemical', 'O', 'B-Disease', 'I-Chemical', 'I-Disease']
        # self.__dict_nen_label = ['X']  # TODO 要遍历完训练集和验证集才能确定__dict_nen_label
        # output = open('./data/dataset_cache/cdr/nen_labels.pkl', 'wb')
        # pickle.dump(self.__dict_nen_label, output, -1)
        # output.close()
        task_name = self.__dict_dataset['train'][:-10].lower() # 'cdr'
        nen_label_pkl_file = open('./data/dataset_cache/' + task_name + '/nen_labels.pkl', 'rb')
        self.__dict_nen_label = pickle.load(nen_label_pkl_file)
        nen_label_pkl_file.close()
        # 对训练集__parse_data后self.__dict_nen_label: ['X', 'D009270', 'O', 'D003000', 'D006973', '-1', 'D007022', ..., , 'D003513', 'D018491', 'D006719']
        self.__max_seq_len = max_len

    def get_train_sample(self):
        return self.__parse_data(f"{BASEPATH}{self.__dict_dataset['train']}")

    def get_eval_sample(self):
        return self.__parse_data(f"{BASEPATH}{self.__dict_dataset['dev']}")
    
    def get_test_sample(self):
        return self.__parse_data(f"{BASEPATH}{self.__dict_dataset['test']}")

    def get_zs_test_sample(self):
        return self.__parse_data(f"{BASEPATH}{self.__dict_dataset['zs_test']}")

    def get_ner_labels(self):
        return self.__dict_ner_label

    def get_nen_labels(self):
        return self.__dict_nen_label

    def __parse_data(self, path: str) -> Dict:
        """
        handle the input data files to the list form
        """
        # 如 path：'./dataset/CDR/train.txt'
        ner_tags = []
        nen_tags = []
        sentences = []
        '''
        ner_tags:
        [] ->
        [['B-Chemical', 'O', 'O', 'O', 'O', 'O', 'B-Chemical', 'O']]->
        ... ->
        [['B-Chemical', 'O', 'O', 'O', 'O', 'O', 'B-Chemical', 'O'], ..., ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']]

        nen_tags:
        [] ->
        [['D009270', 'O', 'O', 'O', 'O', 'O', 'D003000', 'O']] -> 
        ... ->
        [['D009270', 'O', 'O', 'O', 'O', 'O', 'D003000', 'O'], ..., ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']]

        sentences:
        [] ->
        [['Naloxone', 'reverses', 'the', 'antihypertensive', 'effect', 'of', 'clonidine', '.']] ->
        ... ->
        [['Naloxone', 'reverses', 'the', 'antihypertensive', 'effect', 'of', 'clonidine', '.'], ..., ['The', 'implications', 'of', 'these', 'findings', 'are', 'discussed', 'with', 'particular', 'emphasis', 'upon', 'conducting', 'psychopharmacological', 'challenge', 'tests', 'in', 'rodents', '.']]
        '''
        with open(path, "r") as fp:
            ner_tag = []
            nen_tag = []
            sentence = []
            '''
            下面的for循环遍历整个'./dataset/CDR/train.txt'，每遍历一个句子
            ner_tag:
            [] ->
            ['B-Chemical'] ->
            ['B-Chemical', 'O'] ->
            ... ->
            ['B-Chemical', 'O', 'O', 'O', 'O', 'O', 'B-Chemical', 'O'] ->
            [] # 每遍历完一个句子，添加到ner_tags，并清空当前ner_tag

            nen_tag:
            [] ->
            ['D009270'] ->
            ['D009270', 'O'] ->
            ... ->
            ['D009270', 'O', 'O', 'O', 'O', 'O', 'D003000', 'O'] ->
            [] # 每遍历完一个句子，添加到nen_tags，并清空当前nen_tag

            sentence:
            [] ->
            ['Naloxone'] ->
            ['Naloxone', 'reverses'] ->
            ... ->
            ['Naloxone', 'reverses', 'the', 'antihypertensive', 'effect', 'of', 'clonidine', '.'] ->
            [] # 每遍历完一个句子，添加到sentences，并清空当前sentence
            '''
            for line in fp.readlines():
                line = line.strip() # 如'Naloxone	B-Chemical	D009270'  等价于'Naloxone\tB-Chemical\tD009270'

                if line: # 如果line不为空字符串'',表示一个句子没读完
                    word, r_tag, *n_tag = line.split("\t") # 如word:'Naloxone',  r_tag:'B-Chemical',  n_tag:['D009270']
                    sentence.append(word)
                    ner_tag.append(r_tag)
                    nen_tag.append(n_tag[0])
                    # TODO
                    if r_tag not in self.__dict_ner_label:
                        self.__dict_ner_label.append(r_tag) # 遍历*训练集*，生成ner标签集合
                    if n_tag[0] not in self.__dict_nen_label:
                        self.__dict_nen_label.append(n_tag[0]) # 遍历*训练集*，生成nen标签集合
                else: # 如果line为空字符串'',表示读完一个句子
                    sentences.append(copy.deepcopy(sentence)) #深拷贝
                    ner_tags.append(copy.deepcopy(ner_tag))
                    # nen_tags.append(self.__parse_nen_tag(nen_tag)) # TODO
                    nen_tags.append(copy.deepcopy(nen_tag))
                    sentence.clear()  # 如['Naloxone', 'reverses', 'the', 'antihypertensive', 'effect', 'of', 'clonidine', '.'] -> []
                    ner_tag.clear()   # 如['B-Chemical', 'O', 'O', 'O', 'O', 'O', 'B-Chemical', 'O'] -> []
                    nen_tag.clear()   # 如['D009270', 'O', 'O', 'O', 'O', 'O', 'D003000', 'O'] -> []
        # data = {"sentences": sentences, "ner": ner_tags, "nen": nen_tags}
        # {
        # 'sentences': [['Naloxone', 'reverses', 'the', 'antihypertensive', 'effect', 'of', 'clonidine', '.'], ..., ['The', 'implications', 'of', 'these', 'findings', 'are', 'discussed', 'with', 'particular', 'emphasis', 'upon', 'conducting', 'psychopharmacological', 'challenge', 'tests', 'in', 'rodents', '.']] ,
        # 'ner': [['B-Chemical', 'O', 'O', 'O', 'O', 'O', 'B-Chemical', 'O'], ..., ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']], 
        # 'nen': [['D009270', 'O', 'O', 'O', 'O', 'O', 'D003000', 'O'], ..., ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']]
        # }
        parsed_sent_indices = []
        parsed_sent_segments = []
        parsed_sent_attention_masks = []
        parsed_ner_labels = []
        parsed_nen_labels = []
        parsed_real_lens = []
        print("Start parsing {}".format(path))
        for i in trange(len(sentences), ascii=True):
            sentence, ner, nen = [ele[i] for ele in [sentences, ner_tags, nen_tags]] 
            # sentence: ['Naloxone', 'reverses', 'the', 'antihypertensive', 'effect', 'of', 'clonidine', '.']
            # ner: ['B-Chemical', 'O', 'O', 'O', 'O', 'O', 'B-Chemical', 'O']
            # nen: ['D009270', 'O', 'O', 'O', 'O', 'O', 'D003000', 'O']
            (
                sent_indice,
                sent_segment,
                sent_attention_mask,
                t_ner_label,
                t_nen_label,
                sent_len
            ) = self.tokenize_sentence(sentence, ner, nen)
            parsed_sent_indices.append(sent_indice)
            parsed_sent_segments.append(sent_segment)
            parsed_sent_attention_masks.append(sent_attention_mask)
            parsed_ner_labels.append(t_ner_label)
            parsed_nen_labels.append(t_nen_label)
            parsed_real_lens.append(sent_len)
        print("Finish parsing {}".format(path))
        # 对于训练集，parsed_sent_indices: 长为5819的列表，每个元素为一个torch.Size([128])的tensor
        # 对于训练集，parsed_sent_segments: 长为5819的列表，每个元素为一个torch.Size([128])的tensor
        # 对于训练集，parsed_sent_attention_masks: 长为5819的列表，每个元素为一个torch.Size([128])的tensor
        # 对于训练集，parsed_ner_labels: 长为5819的列表，每个元素为一个torch.Size([128])的tensor
        # 对于训练集，parsed_nen_labels: 长为5819的列表，每个元素为一个torch.Size([128])的tensor
        # 对于训练集，parsed_real_lens: 长为5819的列表，每个元素为一个整型，表示该句子的真实长度
        return (parsed_sent_indices, parsed_sent_segments, parsed_sent_attention_masks, parsed_ner_labels, parsed_nen_labels, parsed_real_lens) # 6个元素组成的元组

    def tokenize_sentence(self, sentence, ner_label, nen_label):
        # sentence:['Naloxone', 'reverses', 'the', 'antihypertensive', 'effect', 'of', 'clonidine', '.']
        # ner_label:['B-Chemical', 'O', 'O', 'O', 'O', 'O', 'B-Chemical', 'O']
        # nen_label:['D009270', 'O', 'O', 'O', 'O', 'O', 'D003000', 'O']


        tmp_indice, tmp_segment, tmp_attention_mask, tmp_ner_label, tmp_nen_label = [], [], [], [], []
        s = sentence[: self.__max_seq_len - 2] # ['Naloxone', 'reverses', 'the', 'antihypertensive', 'effect', 'of', 'clonidine', '.']
        CLS_idx, SEP_idx = 0, 0 # 真正的值为CLS_idx:101   SEP_idx:102

        for j, w in enumerate(s): # 如 j:0  w:'Naloxone'
            encoded_word = self.tokenizer(w)
            ind = encoded_word["input_ids"] # self.tokenizer为BertTokenizer 
            seg = encoded_word["token_type_ids"]
            att = encoded_word["attention_mask"]
            # 如ind: [101, 9468, 2858, 21501, 1162, 102] 对应 '[CLS] na ##lo ##xon ##e [SEP]'
            # 如seg: [0, 0, 0, 0, 0, 0]
            # 如att: [1, 1, 1, 1, 1, 1]
            if j == 0:
                CLS_idx, SEP_idx = ind[0], ind[-1] # CLS_idx:101   SEP_idx:102
            ind = ind[1:-1] # [9468, 2858, 21501, 1162]
            seg = seg[1:-1] # [0, 0, 0, 0]
            att = att[1:-1] # [1, 1, 1, 1]

            for k in range(len(ind)): 
                if k == 0:
                    tmp_ner_label.append(self.__dict_ner_label.index(ner_label[j])) 
                    tmp_nen_label.append(self.__dict_nen_label.index(nen_label[j]))
                else:
                    if ner_label[j][0] == 'B':
                        tmp_ner_label.append(self.__dict_ner_label.index('I' + ner_label[j][1:])) 
                    else :
                        tmp_ner_label.append(self.__dict_ner_label.index(ner_label[j])) 
                    # TODO 
                    tmp_nen_label.append(self.__dict_nen_label.index(nen_label[j]))
            # 如对于'Naloxone'对应的ind:[9468, 2858, 21501, 1162], 执行完上面这个for k后tmp_ner_label为[1, 4, 4, 4]，tmp_nen_label为[1, 1, 1, 1]
            # 如对于'reverses'对应的ind:[7936, 1116], 执行完上面这个for k后tmp_ner_label为[1, 4, 4, 4, 2, 2]，tmp_nen_label为[1, 1, 1, 1, 2, 2]
            tmp_indice.extend(ind) # [] -> [9468, 2858, 21501, 1162] -> [9468, 2858, 21501, 1162, 7936, 1116] -> ... -> [9468, 2858, 21501, 1162, 7936, 1116, 1103, 2848, 7889, 17786, 5026, 2109, 2629, 1104, 172, 4934, 2386, 2042, 119]
            tmp_segment.extend(seg)# [] -> [0, 0, 0, 0] -> [0, 0, 0, 0, 0, 0] -> ... -> [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            tmp_attention_mask.extend(att) # [] -> [1, 1, 1, 1] -> [1, 1, 1, 1, 1, 1] -> ... -> [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        # 遍历完一句话，如['Naloxone', 'reverses', 'the', 'antihypertensive', 'effect', 'of', 'clonidine', '.']后，
        # tmp_indice:        [9468, 2858, 21501, 1162, 7936, 1116, 1103, 2848, 7889, 17786, 5026, 2109, 2629, 1104, 172, 4934, 2386, 2042, 119]
        # tmp_segment:       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # tmp_attention_mask:[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        # tmp_ner_label:     [1, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 4, 4, 4, 2]
        # tmp_nen_label:     [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 2]
        tmp_indice = tmp_indice[: self.__max_seq_len - 2]    # [9468, 2858, 21501, 1162, 7936, 1116, 1103, 2848, 7889, 17786, 5026, 2109, 2629, 1104, 172, 4934, 2386, 2042, 119]
        tmp_indice = [CLS_idx] + tmp_indice + [SEP_idx] # [101, 9468, 2858, 21501, 1162, 7936, 1116, 1103, 2848, 7889, 17786, 5026, 2109, 2629, 1104, 172, 4934, 2386, 2042, 119, 102]

        tmp_segment = tmp_segment[: self.__max_seq_len - 2] # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        tmp_segment = [0] + tmp_segment + [0]            # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        tmp_attention_mask = tmp_attention_mask[: self.__max_seq_len - 2] # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        tmp_attention_mask = [1] + tmp_attention_mask + [1]            # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        tmp_ner_label = tmp_ner_label[: self.__max_seq_len - 2]             # [1, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 4, 4, 4, 2]
        tmp_ner_label = (                                                # [2, 1, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 4, 4, 4, 2, 2]
            [self.__dict_ner_label.index("O")] + tmp_ner_label + [self.__dict_ner_label.index("O")]
        )

        tmp_nen_label = tmp_nen_label[: self.__max_seq_len - 2]             # [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 2]
        tmp_nen_label = (                                                # [2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2]
            [self.__dict_ner_label.index("O")] + tmp_nen_label + [self.__dict_ner_label.index("O")]
        )
        List2TensorWithPad = lambda line, max_len, padding_token: torch.tensor(line + [padding_token] * (max_len - len(line)))
        tmp_len = len(tmp_indice)
        tmp_indice = List2TensorWithPad(tmp_indice, self.__max_seq_len, self.tokenizer.vocab['[PAD]']) # 转成tensor且padding
        # tensor([  101,  9468,  2858, 21501,  1162,  7936,  1116,  1103,  2848,  7889,
        #         17786,  5026,  2109,  2629,  1104,   172,  4934,  2386,  2042,   119,
        #           102,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
        #           ...,
        #             0,     0,     0,     0,     0,     0,     0,     0,     0,     0]) # torch.Size([self.__max_seq_len]) 如torch.Size([128])
        tmp_segment = List2TensorWithPad(tmp_segment, self.__max_seq_len, 0)
        # tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        #         ...,
        #         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # torch.Size([self.__max_seq_len]) 
        tmp_attention_mask = List2TensorWithPad(tmp_attention_mask, self.__max_seq_len, 0)
        # tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
        #         ...,
        #         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # torch.Size([self.__max_seq_len]) 
        tmp_ner_label = List2TensorWithPad(tmp_ner_label, self.__max_seq_len, self.__dict_ner_label.index("X"))
        # tensor([2, 1, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 4, 4, 4, 2, 2, 0, 0, 0,
        #         ...,
        #         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # torch.Size([self.__max_seq_len]) 

        tmp_nen_label = List2TensorWithPad(tmp_nen_label, self.__max_seq_len, self.__dict_nen_label.index("X"))
        # tensor([2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 0, 0, 0,
        #         ...,
        #         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # torch.Size([self.__max_seq_len])

        return (tmp_indice, tmp_segment, tmp_attention_mask, tmp_ner_label, tmp_nen_label, tmp_len) # 返回由5个大小为torch.Size([self.__max_seq_len]) 的tensor 和 1个int 组成的元组

    # def __parse_nen_tag(self, nen_tag): 
        # 修正后的
        #     """
        #     parse the nen tag sequence to the dictionary 将nen标签序列解析为字典
        #     example:
        #         nen_tag:['D009270', 'D009270', 'D009080', 'O', 'O', 'D009080', 'D009080', 'O', 'O', 'D003000', 'O', 'D003000', 'D003000'] -> result:{'D009270': [(0, 2)], 'D009080': [(2, 1), (5, 2)], 'D003000': [(9, 1), (11, 2)]}
        #     """
        #     tmp = []
        #     result = {}

        #     for i in range(len(nen_tag)):
        #         if nen_tag[i] != "O" :
        #             if(i==0): tmp.append(i)
        #             else :
        #                 if(nen_tag[i-1]=="O") : tmp.append(i)
        #                 else :
        #                     if(nen_tag[i]==nen_tag[i-1]) : tmp.append(i)
        #                     else:
        #                         if len(tmp):
        #                             cache = result.get(nen_tag[i-1], [])
        #                             cache.append((tmp[0], len(tmp)))
        #                             result[nen_tag[i - 1]] = cache
        #                             tmp.clear()
        #                             tmp.append(i)
        #         else:
        #             if len(tmp):
        #                 cache = result.get(nen_tag[i-1], [])
        #                 cache.append((tmp[0], len(tmp)))
        #                 result[nen_tag[i - 1]] = cache
        #                 tmp.clear()
        #     if len(tmp):
        #         i = len(nen_tag)
        #         cache = result.get(nen_tag[i-1], [])
        #         cache.append((tmp[0], len(tmp)))
        #         result[nen_tag[i - 1]] = cache
        #         tmp.clear()
        #     return result


class NCBIProcessor(object):
    def __init__(self, tokenizer, dict_dataset: Dict, ):
        pass

if __name__ == '__main__':
    # 测试Processor
    DICT_DATASET = {
                "train": "CDR/train.txt",
                "dev": "CDR/dev.txt",
                "test": "CDR/test.txt",
                "zs_test": "CDR/zs_test.txt",
                }

    bert_path = '/Users/wangyang/Desktop/wy/project/python/deeplearning/nernen/data/model_data/biobert-base-cased-v1.2' # TODO 测试用，在服务器上部署时要换成biobert-large-cased-v1.1
    from transformers import BertTokenizer, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(bert_path) # 分词器
    p = Processor(tokenizer, DICT_DATASET)
    train_samples = p.get_train_sample()
    print(train_samples)

    # 测试分词方法tokenize_sentence
    sentence = ['Naloxone', 'reverses', 'the', 'antihypertensive', 'effect', 'of', 'clonidine', '.']
    ner_label = ['B-Chemical', 'O', 'O', 'O', 'O', 'O', 'B-Chemical', 'O']
    nen_label = ['D009270', 'O', 'O', 'O', 'O', 'O', 'D003000', 'O']
    print(p.tokenize_sentence(sentence, ner_label, nen_label))


    '''
    # 测试Bert分词器
    print(p.tokenizer(["good morning", "What a Great World!", "What a Great World", "你好！"]))
    # {'input_ids': [[101, 1363, 2106, 102], [101, 1184, 170, 1632, 1362, 106, 102], [101, 1184, 170, 1632, 1362, 102], [101, 100, 100, 1096, 102]], 'token_type_ids': [[0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]}

    print(p.tokenizer.decode(tokenizer("What a Great World!")['input_ids']))
    # [CLS] what a great world! [SEP]

    print(p.tokenizer.decode(tokenizer("你好！")['input_ids']))
    # [CLS] [UNK] [UNK] ！ [SEP]
    '''


# class DataLoader():
#     def __init__(self, bert_path: str, dict_dataset: Dict):
#         self.__bert_path = f"{bert_path}/vocab.txt"
#         self.__tokenizer = self.__load_vocabulary()
#         self.__dict_dataset = dict_dataset
#         self.__entity_base = EntityBase(bert_path)
#         self.__dict_ner_label = ["X"]
#         # 执行完self.__get_sentences('./dataset/CDR/train.txt')后 self.__dict_ner_label:['X', 'B-Chemical', 'O', 'B-Disease', 'I-Chemical', 'I-Disease']

#         self.__max_seq_len = 100
#         self.__max_ent_len = 16
#         self.__batch_size = 6
#         self._train_data = self.__parse_data(f"{BASEPATH}{dict_dataset['train']}")
#         print(222222)
#         self._devel_data = self.__parse_data(f"{BASEPATH}{dict_dataset['dev']}")
#         print(222233)
#         self._test_data = self.__parse_data(f"{BASEPATH}{dict_dataset['test']}")
#         print(2222344)
#         self._zs_test_data = self.__parse_data(f"{BASEPATH}{dict_dataset['zs_test']}")

#     def resampling_data(self, dtype):
#         if dtype == "train":
#             self._train_data = self.__parse_data(
#                 f"{BASEPATH}{self.__dict_dataset['train']}"
#             )
#         elif dtype == "devel":
#             self._devel_data = self.__parse_data(
#                 f"{BASEPATH}{self.__dict_dataset['dev']}"
#             )
#         elif dtype == "test":
#             self._test_data = self.__parse_data(
#                 f"{BASEPATH}{self.__dict_dataset['test']}"
#             )

#     def parse_idx_tokens(self, ind):
#         return self.__tokenizer.decode(ind)

#     def parse_idx_ner_labels(self, ner):
#         return [self.__dict_ner_label[i] for i in ner]

#     def __parse_idx_sequence(self, pred, label):
#         res_pred, res_label = [], []
#         records = {}

#         for i in range(len(pred)):
#             tmp_pred, tmp_label = [], []
#             str_pred = " ".join([str(ele) for ele in pred[i].numpy().tolist()])
#             str_label = " ".join([str(ele) for ele in label[i].numpy().tolist()])
#             str_record = str_pred + str_label
#             if str_record in records:
#                 tmp = records[str_record]
#                 res_pred.append(tmp[0])
#                 res_label.append(tmp[1])

#             else:
#                 for p, l in zip(pred[i], label[i]):
#                     if self.__dict_ner_label[l] != "X":
#                         tmp_label.append(self.__dict_ner_label[l])

#                         if self.__dict_ner_label[p] == "X":
#                             tmp_pred.append("O")
#                         else:
#                             tmp_pred.append(self.__dict_ner_label[p])
#                 res_pred.append(tmp_pred)
#                 res_label.append(tmp_label)
#                 records[str_record] = (tmp_pred, tmp_label)

#         return res_pred, res_label

#     def evaluate_ner(self, logits, label, real_len):
#         pred = tf.argmax(logits, axis=-1)
#         pred, true = self.__parse_idx_sequence(pred, label)
#         y_real, pred_real = [], []
#         records = []
#         for i in trange(len(real_len), ascii=True):
#             record = " ".join(true[i]) + str(real_len[i])
#             if record not in records:
#                 records.append(record)
#                 y_real.extend(true[i][1 : 1 + real_len[i]])
#                 pred_real.extend(pred[i][1 : 1 + real_len[i]])
#         prec, rec, f1 = evaluate(y_real, pred_real, verbose=False)
#         return (prec / 100, rec / 100, f1 / 100)

#     def __restore_ner_label(self, ner_logits, ner_label, real_len):
#         ner_pred = tf.argmax(ner_logits, axis=-1)
#         ner_pred, ner_truth = self.__parse_idx_sequence(ner_pred, ner_label)
#         ner_label_real, ner_pred_real = [], []
#         for i in range(len(real_len)):
#             ner_label_real.append(ner_truth[i][1 : 1 + real_len[i]])
#             ner_pred_real.append(ner_pred[i][1 : 1 + real_len[i]])
#         return ner_label_real, ner_pred_real

#     def __extract_entity_by_index(self, label_sequence, index):
#         length = len(label_sequence)
#         entity = []
#         tmp_index = index
#         while (
#             tmp_index >= 0 and tmp_index < length and label_sequence[tmp_index] != "O"
#         ):
#             entity.insert(0, tmp_index)
#             if "B-" in label_sequence[tmp_index]:
#                 break
#             tmp_index -= 1
#         tmp_index = index + 1
#         while tmp_index < length and label_sequence[tmp_index] != "O":
#             if "B-" in label_sequence[tmp_index]:
#                 break
#             entity.append(tmp_index)
#             tmp_index += 1
#         return entity

#     def evaluate_nen(
#         self,
#         ner_logits,
#         ner_label,
#         cpt_ner_logits,
#         cpt_ner_label,
#         real_len,
#         nen_logits,
#         nen_label,
#     ):
#         ner_label_real, ner_pred_real = self.__restore_ner_label(
#             ner_logits, ner_label, real_len
#         )
#         cpt_ner_label_real, cpt_ner_pred_real = self.__restore_ner_label(
#             cpt_ner_logits, cpt_ner_label, real_len
#         )

#         nen_pred = tf.argmax(nen_logits, axis=-1).numpy().tolist()
#         nen_label = nen_label.numpy().tolist()
#         tmp_nen_pred = []
#         tmp_nen_label = []
#         for i in trange(len(nen_label), ascii=True):
#             n_entity = 0
#             if nen_label[i] == 1:
#                 for e in ner_label_real:
#                     if "B-" in e:
#                         n_entity += 1
#                 tmp_nen_label.extend([1] * n_entity)
#             else:
#                 tmp_nen_label.append(0)

#             if nen_pred[i] == 0:
#                 tmp_nen_pred.extend([0] * n_entity)
#             else:
#                 index = 0
#                 flag = False
#                 for p, t in zip(ner_pred_real[i], ner_label_real[i]):
#                     if "B-" in p or "I-" in p:
#                         if p == t:
#                             if not flag:
#                                 if self.__extract_entity_by_index(
#                                     cpt_ner_pred_real[i], index
#                                 ) == self.__extract_entity_by_index(
#                                     cpt_ner_label_real[i], index
#                                 ):
#                                     tmp_nen_pred.append(1)
#                                 else:
#                                     tmp_nen_pred.append(0)
#                                 flag = True
#                         else:
#                             if not flag:
#                                 tmp_nen_pred.append(0)
#                                 flag = True
#                     else:
#                         flag = False
#                     index += 1

#             if len(tmp_nen_label) < len(tmp_nen_pred):
#                 size = len(tmp_nen_label)
#                 for _ in range(len(tmp_nen_pred) - size):
#                     tmp_nen_label.append(nen_label[i])
#             elif len(tmp_nen_label) > len(tmp_nen_pred):
#                 size = len(tmp_nen_pred)
#                 for _ in range(len(tmp_nen_label) - size):
#                     tmp_nen_pred.append(0)

#         filtered_nen_label, filtered_nen_pred = [], []
#         for i in range(len(tmp_nen_label)):
#             if tmp_nen_label[i] == 0 and tmp_nen_pred[i] == 0:
#                 continue
#             filtered_nen_label.append(tmp_nen_label[i])
#             filtered_nen_pred.append(tmp_nen_pred[i])
#         reca = recall_score(filtered_nen_label, filtered_nen_pred, average="weighted")
#         prec = precision_score(
#             filtered_nen_label, filtered_nen_pred, average="weighted"
#         )
#         f1 = f1_score(filtered_nen_label, filtered_nen_pred, average="weighted")
#         return (prec, reca, f1)

#     @property
#     def LABEL_SIZE(self):
#         return len(self.__dict_ner_label)

#     def Data(self, dtype: str):
#         return getattr(self, f"_{dtype}_data")

#     def __load_vocabulary(self):
#         token_dict = {}

#         with codecs.open(self.__bert_path, "r", "utf8") as reader: # self.__bert_path: f"{bert_path}/vocab.txt"
#             for line in reader:
#                 token = line.strip()
#                 token_dict[token] = len(token_dict)

#         return Tokenizer(token_dict) # keras_bert.Tokenizer

#     def __tokenize_entity(self, entities):
#         indices = []
#         segments = []

#         for e in entities:
#             ind, seg = self.__tokenizer.encode(first=e)
#             indices.append(ind)
#             segments.append(seg)

#         indices = pad_sequences(indices, self.__max_ent_len, value=0, padding="post")
#         segments = pad_sequences(segments, self.__max_ent_len, value=0, padding="post")

#         return (indices, segments)

#     def __tokenize_sample(self, sentence, label, cpt_label):
#         # sentence:[['Naloxone', 'reverses', 'the', 'antihypertensive', 'effect', 'of', 'clonidine', '.'], ['Naloxone', 'reverses', 'the', 'antihypertensive', 'effect', 'of', 'clonidine', '.'], ['Naloxone', 'reverses', 'the', 'antihypertensive', 'effect', 'of', 'clonidine', '.']]
#         # label:[['B-Chemical', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['O', 'O', 'O', 'O', 'O', 'O', 'B-Chemical', 'O'], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']]
#         # cpt_label:[['B-Chemical', 'O', 'O', 'O', 'O', 'O', 'B-Chemical', 'O'], ['B-Chemical', 'O', 'O', 'O', 'O', 'O', 'B-Chemical', 'O'], ['B-Chemical', 'O', 'O', 'O', 'O', 'O', 'B-Chemical', 'O']]
#         indices = []
#         segments = []
#         labels = []
#         cpt_labels = []

#         for i in range(len(sentence)):
#             s = sentence[i][: self.__max_seq_len - 2] # ['Naloxone', 'reverses', 'the', 'antihypertensive', 'effect', 'of', 'clonidine', '.']
#             tmp_indice, tmp_segment, tmp_label, tmp_cpt_label = [], [], [], []
#             CLS_idx, SEP_idx = 0, 0

#             for j, w in enumerate(s): # 如 j:0  w:'Naloxone'
#                 ind, seg = self.__tokenizer.encode(first=w) # self.__tokenizer为self.__load_vocabulary()返回的keras_bert.Tokenizer(token_dict)  其中token_dict从"{bert_path}/vocab.txt"中获取（token2id）
#                 # 如ind: [101, 29420, 10649, 4798, 102]  seg:[0, 0, 0, 0, 0]
#                 if j == 0:
#                     CLS_idx, SEP_idx = ind[0], ind[-1] # CLS_idx:101   SEP_idx:102
#                 ind = ind[1:-1] # [29420, 10649, 4798]
#                 seg = seg[1:-1] # [0, 0, 0]

#                 for k in range(len(ind)): 
#                     if k == 0:
#                         tmp_label.append(self.__dict_ner_label.index(label[i][j]))
#                         tmp_cpt_label.append(
#                             self.__dict_ner_label.index(cpt_label[i][j])
#                         )
#                     else:
#                         tmp_label.append(self.__dict_ner_label.index("X"))
#                         tmp_cpt_label.append(
#                             self.__dict_ner_label.index(cpt_label[i][j])
#                         )
#                 # 如对于'Naloxone'对应的ind:[29420, 10649, 4798], 执行完上面这个for k后tmp_label为[1, 0, 0]，tmp_cpt_label为[1, 1, 1]
#                 # 如对于'reverses'对应的ind:[7936, 1116], 执行完上面这个for k后tmp_label为[1, 0, 0, 2, 0]，tmp_cpt_label为[1, 1, 1, 2, 2]
#                 tmp_indice.extend(ind) # [] -> [29420, 10649, 4798] -> [29420, 10649, 4798, 7936, 1116] -> ... -> [29420, 10649, 4798, 7936, 1116, 1103, 2848, 41624, 2629, 1104, 29260, 11153, 10399, 119]
#                 tmp_segment.extend(seg)# [] -> [0, 0, 0] -> [0, 0, 0, 0, 0] -> ... -> [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

#             # 遍历完一句话，如['Naloxone', 'reverses', 'the', 'antihypertensive', 'effect', 'of', 'clonidine', '.']后，
#             # tmp_indice:[29420, 10649, 4798, 7936, 1116, 1103, 2848, 41624, 2629, 1104, 29260, 11153, 10399, 119]
#             # tmp_segment:  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#             # tmp_label:    [1, 0, 0, 2, 0, 2, 2, 0, 2, 2, 2, 0, 0, 2]
#             # tmp_cpt_label:[1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2]
#             tmp_indice = tmp_indice[: self.__max_seq_len - 2] # [29420, 10649, 4798, 7936, 1116, 1103, 2848, 41624, 2629, 1104, 29260, 11153, 10399, 119]
#             tmp_indice = [CLS_idx] + tmp_indice + [SEP_idx] # [101, 29420, 10649, 4798, 7936, 1116, 1103, 2848, 41624, 2629, 1104, 29260, 11153, 10399, 119, 102]

#             tmp_segment = tmp_segment[: self.__max_seq_len - 2] # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#             tmp_segment = [0] + tmp_segment + [0]            # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

#             tmp_label = tmp_label[: self.__max_seq_len - 2]             # [1, 0, 0, 2, 0, 2, 2, 0, 2, 2, 2, 0, 0, 2]
#             tmp_label = (                                            # [2, 1, 0, 0, 2, 0, 2, 2, 0, 2, 2, 2, 0, 0, 2, 2]
#                 [self.__dict_ner_label.index("O")]
#                 + tmp_label
#                 + [self.__dict_ner_label.index("O")]
#             )

#             tmp_cpt_label = tmp_cpt_label[: self.__max_seq_len - 2]     # [1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2]
#             tmp_cpt_label = (                                        # [2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2]
#                 [self.__dict_ner_label.index("O")]
#                 + tmp_cpt_label
#                 + [self.__dict_ner_label.index("O")]
#             )

#             indices.append(tmp_indice)
#             segments.append(tmp_segment)
#             labels.append(tmp_label)
#             cpt_labels.append(tmp_cpt_label)
#             '''
#             indices:
#             [] ->
#             [[101, 29420, 10649, 4798, 7936, 1116, 1103, 2848, 41624, 2629, 1104, 29260, 11153, 10399, 119, 102]] ->
#             [[101, 29420, 10649, 4798, 7936, 1116, 1103, 2848, 41624, 2629, 1104, 29260, 11153, 10399, 119, 102], [101, 29420, 10649, 4798, 7936, 1116, 1103, 2848, 41624, 2629, 1104, 29260, 11153, 10399, 119, 102]] ->
#             [[101, 29420, 10649, 4798, 7936, 1116, 1103, 2848, 41624, 2629, 1104, 29260, 11153, 10399, 119, 102], [101, 29420, 10649, 4798, 7936, 1116, 1103, 2848, 41624, 2629, 1104, 29260, 11153, 10399, 119, 102], [101, 29420, 10649, 4798, 7936, 1116, 1103, 2848, 41624, 2629, 1104, 29260, 11153, 10399, 119, 102]]

#             segments:
#             [] ->
#             [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] ->
#             [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] ->
#             [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

#             labels:
#             [] ->
#             [[2, 1, 0, 0, 2, 0, 2, 2, 0, 2, 2, 2, 0, 0, 2, 2]] ->
#             [[2, 1, 0, 0, 2, 0, 2, 2, 0, 2, 2, 2, 0, 0, 2, 2], [2, 2, 0, 0, 2, 0, 2, 2, 0, 2, 2, 1, 0, 0, 2, 2]] ->
#             [[2, 1, 0, 0, 2, 0, 2, 2, 0, 2, 2, 2, 0, 0, 2, 2], [2, 2, 0, 0, 2, 0, 2, 2, 0, 2, 2, 1, 0, 0, 2, 2], [2, 2, 0, 0, 2, 0, 2, 2, 0, 2, 2, 2, 0, 0, 2, 2]]

#             cpt_labels:
#             [] ->
#             [[2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2]] ->
#             [[2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2], [2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2]] ->
#             [[2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2], [2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2], [2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2]]
#             '''

#         labels = pad_sequences( # keras.preprocessing.sequence.pad_sequences
#             labels,
#             self.__max_seq_len, # self.__max_seq_len = 100
#             value=self.__dict_ner_label.index("O"),
#             padding="post",
#         )
#         # labels: 大小为(3,100)的ndarray 
#         # array([[ 2, 1, 0, 0, 2, 0, 2, 2, 0, 2, 2, 2, 0, 0, 2, 2, 2., 2, 2, ..., 2, 2, 2], # 每行长100 从2.开始为padding
#         #        [ 2, 2, 0, 0, 2, 0, 2, 2, 0, 2, 2, 1, 0, 0, 2, 2, 2., 2, 2, ..., 2, 2, 2],
#         #        [ 2, 2, 0, 0, 2, 0, 2, 2, 0, 2, 2, 2, 0, 0, 2, 2, 2., 2, 2, ..., 2, 2, 2]])
#         cpt_labels = pad_sequences(
#             cpt_labels,
#             self.__max_seq_len, 
#             value=self.__dict_ner_label.index("O"),
#             padding="post",
#         )
#         # cpt_labels: 大小为(3,100)的ndarray 
#         # array([[2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2., 2, 2, ..., 2, 2, 2], # 每行长100 从2.开始为padding
#         #        [2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2., 2, 2, ..., 2, 2, 2],
#         #        [2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2., 2, 2, ..., 2, 2, 2]])
#         indices = pad_sequences(indices, self.__max_seq_len, value=0, padding="post")
#         # indices: 大小为(3,100)的ndarray 
#         # array([[101, 29420, 10649, 4798, 7936, 1116, 1103, 2848, 41624, 2629, 1104, 29260, 11153, 10399, 119, 102, 0, 0, 0, ..., 0, 0, 0], # 每行长100 从0开始为padding
#         #        [101, 29420, 10649, 4798, 7936, 1116, 1103, 2848, 41624, 2629, 1104, 29260, 11153, 10399, 119, 102, 0, 0, 0, ..., 0, 0, 0], 
#         #        [101, 29420, 10649, 4798, 7936, 1116, 1103, 2848, 41624, 2629, 1104, 29260, 11153, 10399, 119, 102, 0, 0, 0, ..., 0, 0, 0]])
#         segments = pad_sequences(segments, self.__max_seq_len, value=0, padding="post")
#         # segments: 大小为(3,100)的ndarray  元素全为0

#         return (indices, segments, labels, cpt_labels) # 返回由4个大小为(3,100)的ndarray 组成的元组

#     def __parse_data(self, path: str):
#         data = self.__get_sentences(path)
#         # {
#         # 'sentences': [['Naloxone', 'reverses', 'the', 'antihypertensive', 'effect', 'of', 'clonidine', '.'], ..., ['The', 'implications', 'of', 'these', 'findings', 'are', 'discussed', 'with', 'particular', 'emphasis', 'upon', 'conducting', 'psychopharmacological', 'challenge', 'tests', 'in', 'rodents', '.']] ,
#         # 'ner': [['B-Chemical', 'O', 'O', 'O', 'O', 'O', 'B-Chemical', 'O'], ..., ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']], 
#         # 'nen': [{'D009270': [(0, 1)], 'D003000': [(6, 1)]}, ..., {'D001058': [(2, 1)], 'D004298': [(8, 1)]}, {}]}
#         # }
#         sentences = data["sentences"]
#         ner_label = data["ner"]
#         nen_label = data["nen"]

#         parsed_sent_indices = []
#         parsed_sent_segments = []
#         parsed_ner_label = []
#         parsed_cpt_ner_label = []
#         parsed_real_len = []
#         parsed_nen_label = []
#         parsed_ent_indices = []
#         parsed_ent_segments = []

#         for i in trange(len(sentences), ascii=True):
#             sentence, ner, nen = [ele[i] for ele in [sentences, ner_label, nen_label]] 
#             # sentence: ['Naloxone', 'reverses', 'the', 'antihypertensive', 'effect', 'of', 'clonidine', '.']
#             # ner: ['B-Chemical', 'O', 'O', 'O', 'O', 'O', 'B-Chemical', 'O']
#             # nen: {'D009270': [(0, 1)], 'D003000': [(6, 1)]}
#             samples = self.__extend_sample(sentence, ner, nen, 1, "test" in path)
#             # {'ner': [['B-Chemical', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['O', 'O', 'O', 'O', 'O', 'O', 'B-Chemical', 'O'], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']], 
#             # 'cpt_ner': [['B-Chemical', 'O', 'O', 'O', 'O', 'O', 'B-Chemical', 'O'], ['B-Chemical', 'O', 'O', 'O', 'O', 'O', 'B-Chemical', 'O'], ['B-Chemical', 'O', 'O', 'O', 'O', 'O', 'B-Chemical', 'O']], 
#             # 'nen': [1, 1, 0], 
#             # 'sentences': [['Naloxone', 'reverses', 'the', 'antihypertensive', 'effect', 'of', 'clonidine', '.'], ['Naloxone', 'reverses', 'the', 'antihypertensive', 'effect', 'of', 'clonidine', '.'], ['Naloxone', 'reverses', 'the', 'antihypertensive', 'effect', 'of', 'clonidine', '.']], 
#             # 'ent_sents': ['Naloxone', 'Clonidine', 'Reperfusion Injury']}
#             pkd_sentence = samples["sentences"]
#             pkd_ner_tags = samples["ner"]
#             pkd_cpt_ner_tags = samples["cpt_ner"]
#             pkd_nen_tags = samples["nen"]
#             pkd_ent_sents = samples["ent_sents"]

#             (
#                 sent_indices,
#                 sent_segments,
#                 t_ner_label,
#                 t_cpt_ner_label,
#             ) = self.__tokenize_sample(pkd_sentence, pkd_ner_tags, pkd_cpt_ner_tags)
#             parsed_sent_indices.append(sent_indices)
#             parsed_sent_segments.append(sent_segments)
#             parsed_ner_label.append(t_ner_label)
#             parsed_cpt_ner_label.append(t_cpt_ner_label)

#             parsed_nen_label.extend(pkd_nen_tags)
#             ent_indices, ent_segments = self.__tokenize_entity(pkd_ent_sents) 
#             # ent_indices:  array([[101, 29420, 10649, 4798, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 29260, 11153, 10399, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 55281, 3773, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
#             # ent_segments: array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
#             parsed_ent_indices.append(ent_indices)
#             parsed_ent_segments.append(ent_segments)

#             parsed_real_len.extend([len(sentence)] * len(sent_indices))
#         '''
#         parsed_real_len: 
#         [] -> 
#         [8, 8, 8] ->
#         ... ->
#         [8, 8, 8, 35, 35, 35, 35, 7, 18, 18, 18, 18, 12, ... ] # 长度为14139的列表

#         parsed_ner_label: 
#         [] -> 
#         [array([[ 2, 1, 0, 0, 2, 0, 2, 2, 0, 2, 2, 2, 0, 0, 2, 2, 2., 2, 2, ..., 2, 2, 2], [ 2, 2, 0, 0, 2, 0, 2, 2, 0, 2, 2, 1, 0, 0, 2, 2, 2., 2, 2, ..., 2, 2, 2], [ 2, 2, 0, 0, 2, 0, 2, 2, 0, 2, 2, 2, 0, 0, 2, 2, 2., 2, 2, ..., 2, 2, 2]])] ->
#         ... ->
#         长为5819的列表 ，每个元素是一个ndarray, 这个ndarray大小为(*, 100)

#         parsed_cpt_ner_label: 
#         [] -> 
#         [array([[2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2., 2, 2, ..., 2, 2, 2], [2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2., 2, 2, ..., 2, 2, 2], [2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2., 2, 2, ..., 2, 2, 2]])] ->
#         ... ->
#         长为5819的列表 ，每个元素是一个ndarray, 这个ndarray大小为(*, 100)

#         parsed_nen_label: 
#         [] -> 
#         [1, 1, 0] ->
#         ... ->
#         [1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, ... ] # 长度为14139的列表

#         parsed_sent_indices: 
#         [] -> 
#         [array([[101, 29420, 10649, 4798, 7936, 1116, 1103, 2848, 41624, 2629, 1104, 29260, 11153, 10399, 119, 102, 0, 0, 0, ..., 0, 0, 0], [101, 29420, 10649, 4798, 7936, 1116, 1103, 2848, 41624, 2629, 1104, 29260, 11153, 10399, 119, 102, 0, 0, 0, ..., 0, 0, 0], [101, 29420, 10649, 4798, 7936, 1116, 1103, 2848, 41624, 2629, 1104, 29260, 11153, 10399, 119, 102, 0, 0, 0, ..., 0, 0, 0]])] ->
#         ... ->
#         长为5819的列表 ，每个元素是一个ndarray, 这个ndarray大小为(*, 100)

#         parsed_sent_segments: 
#         [] -> 
#         [一个元素全为0的大小为(3,100)的ndarray] ->
#         ... ->
#         长为5819的列表 ，每个元素是一个ndarray, 这个ndarray大小为(*, 100)

#         parsed_ent_indices: 
#         [] -> 
#         [array([[101, 29420, 10649, 4798, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 29260, 11153, 10399, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 55281, 3773, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])] ->
#         ... ->
#         长为5819的列表 ，每个元素是一个ndarray, 这个ndarray大小为(*, 16)


#         parsed_ent_segments: 
#         [] -> 
#         [array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])] ->
#         ... ->
#         长为5819的列表 ，每个元素是一个ndarray, 这个ndarray大小为(*, 16)
#         '''

#         parsed_ner_label = np.vstack(parsed_ner_label) # 大小为(14139,100)的ndarray
#         parsed_cpt_ner_label = np.vstack(parsed_cpt_ner_label) # 大小为(14139,100)的ndarray
#         parsed_sent_indices = np.vstack(parsed_sent_indices) # 大小为(14139,100)的ndarray
#         parsed_sent_segments = np.vstack(parsed_sent_segments) # 大小为(14139,100)的ndarray
#         parsed_ent_indices = np.vstack(parsed_ent_indices) # 大小为(14139,16)的ndarray
#         parsed_ent_segments = np.vstack(parsed_ent_segments) # 大小为(14139,16)的ndarray

#         dataset = Dataset.from_tensor_slices( # tensorflow.data.Dataset
#             (
#                 parsed_sent_indices,
#                 parsed_sent_segments,
#                 parsed_ent_indices,
#                 parsed_ent_segments,
#                 parsed_ner_label,
#                 parsed_cpt_ner_label,
#                 parsed_nen_label,
#                 parsed_real_len,
#             )
#         )
#         # <TensorSliceDataset shapes: ((100,), (100,), (16,), (16,), (100,), (100,), (), ()), types: (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32)>
#         dataset = (
#             dataset.shuffle(len(parsed_real_len))
#             .batch(self.__batch_size)
#             .prefetch(tf.data.experimental.AUTOTUNE)
#             .cache()
#         )
#         # <CacheDataset shapes: ((None, 100), (None, 100), (None, 16), (None, 16), (None, 100), (None, 100), (None,), (None,)), types: (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32)>
#         return dataset

#     def __extend_sample(
#         self,
#         sentence: List[int],
#         ner_tag: List[int],
#         nen_tag: Dict,
#         n_neg: int,
#         flag: bool = False,
#     ) -> Dict:
#         """
#         extend the sample to all samples with specific nen tags
#         """
#         pkd_ner_tags = []
#         pkd_nen_tags = []
#         pkd_sentences = []
#         pkd_ent_sents = []
#         pkd_cpt_ner_tags = []

#         for src_ent, _ in nen_tag.items():
#             tmp = ner_tag.copy()
#             pkd_cpt_ner_tags.append(ner_tag.copy())

#             for tar_ent, tar_idxs in nen_tag.items():
#                 if tar_ent != src_ent:
#                     for begin, length in tar_idxs:
#                         for i in range(begin, begin + length):
#                             tmp[i] = "O"

#             pkd_ner_tags.append(tmp)
#             pkd_sentences.append(sentence.copy())

#             ent_sent = self.__entity_base.getItem(src_ent)
#             pkd_ent_sents.append(ent_sent)
#             pkd_nen_tags.append(1)
#         # Sampling negative entities
#         if flag:
#             cands = self.__entity_base.generate_candidates(
#                 sentence, list(nen_tag.keys())
#             )
#             for c in cands:
#                 pkd_ner_tags.append(["O"] * len(ner_tag))
#                 pkd_cpt_ner_tags.append(ner_tag.copy())
#                 pkd_sentences.append(sentence.copy())
#                 ent_sent = self.__entity_base.getItem(c)
#                 pkd_ent_sents.append(ent_sent)
#                 pkd_nen_tags.append(0)
#         else:
#             for i in range(n_neg):
#                 pkd_ner_tags.append(["O"] * len(ner_tag))
#                 pkd_cpt_ner_tags.append(ner_tag.copy())
#                 pkd_sentences.append(sentence.copy())
#                 ent_sent = self.__entity_base.random_entity(list(nen_tag.keys()))
#                 pkd_ent_sents.append(ent_sent)
#                 pkd_nen_tags.append(0)

#         return {
#             "ner": pkd_ner_tags,
#             "cpt_ner": pkd_cpt_ner_tags,
#             "nen": pkd_nen_tags,
#             "sentences": pkd_sentences,
#             "ent_sents": pkd_ent_sents,
#         }

#     def __parse_nen_tag(self, nen_tag: List[str]) -> Dict:
#         """
#         parse the nen tag sequence to the dictionary
#         example:
#             [1,1,-1,-1] -> {
#                 1: [(0,2)]
#             }
#         """
#         tmp = []
#         result = {}

#         for i in range(len(nen_tag)):
#             if nen_tag[i] != "O":
#                 tmp.append(i)
#             else:
#                 if len(tmp):
#                     cache = result.get(nen_tag[i], [])
#                     cache.append((tmp[0], len(tmp)))
#                     result[nen_tag[i - 1]] = cache
#                     tmp.clear()

#         return result

#     def __get_sentences(self, path: str) -> Dict:
#         """
#         handle the input data files to the list form
#         """
#         ner_tags = []
#         nen_tags = []
#         sentences = []
# '''
# ner_tags:
# [] ->
# [['B-Chemical', 'O', 'O', 'O', 'O', 'O', 'B-Chemical', 'O']]->
# ... ->
# [['B-Chemical', 'O', 'O', 'O', 'O', 'O', 'B-Chemical', 'O'], ..., ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']]

# nen_tags:
# [] ->
# [{'D009270': [(0, 1)], 'D003000': [(6, 1)]}] -> 
# ... ->
# [{'D009270': [(0, 1)], 'D003000': [(6, 1)]}, ..., {}]

# sentences:
# [] ->
# [['Naloxone', 'reverses', 'the', 'antihypertensive', 'effect', 'of', 'clonidine', '.']] ->
# ... ->
# [['Naloxone', 'reverses', 'the', 'antihypertensive', 'effect', 'of', 'clonidine', '.'], ..., ['The', 'implications', 'of', 'these', 'findings', 'are', 'discussed', 'with', 'particular', 'emphasis', 'upon', 'conducting', 'psychopharmacological', 'challenge', 'tests', 'in', 'rodents', '.']]
# '''
#         with open(path, "r") as fp:
#             ner_tag = []
#             nen_tag = []
#             sentence = []

#             for line in fp.readlines():
#                 line = line.strip()

#                 if line:
#                     word, r_tag, *n_tag = line.split("\t")
#                     sentence.append(word)
#                     ner_tag.append(r_tag)
#                     nen_tag.append(n_tag[0])

#                     if "train" in path and r_tag not in self.__dict_ner_label:
#                         self.__dict_ner_label.append(r_tag)
#                 else:
#                     sentences.append(copy.deepcopy(sentence))
#                     ner_tags.append(copy.deepcopy(ner_tag))
#                     nen_tags.append(self.__parse_nen_tag(nen_tag))
#                     sentence.clear()
#                     ner_tag.clear()
#                     nen_tag.clear()

#         return {"sentences": sentences, "ner": ner_tags, "nen": nen_tags}
