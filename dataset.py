#!python
# coding=utf-8
'''
:Description  : To be filled
:Version      : 1.0
:Author       : LilNeo
:Date         : 2022-03-17 17:17:37
:LastEditors  : wy
:LastEditTime : 2022-03-23 00:04:21
:FilePath     : /nernen/dataset.py
:Copyright 2022 LilNeo, All Rights Reserved. 
'''
from torch.utils.data import Dataset
import os
import torch
class NERNENDataset(Dataset): # 继承torch.utils.data.Dataset
    def __init__(
            self,
            samples,
            mode='train',
    ):
        super(NERNENDataset, self).__init__()
        '''
        train_samples：parsed_sent_indices, parsed_sent_segments, parsed_sent_attention_masks, parsed_ner_labels, parsed_nen_labels, parsed_real_lens 6个元素组成的元组,详情见data_process.py中Processor类的__parse_data方法
        '''
        self.sentences = [{'input_ids': samples[0][i], 'token_type_ids': samples[1][i], 'attention_mask': samples[2][i]} for i in range(len(samples[0]))]
        self.ner = samples[3]
        self.nen = samples[4]
        self.len = samples[5]
        self.mode = mode

    def __getitem__(self, item):
        if self.mode != 'test': # train和eval模式
            return self.sentences[item], self.ner[item], self.nen[item], self.len[item]
        else: # test模式
            return self.sentences[item], self.len[item]

    def __len__(self):
        return len(self.sentences)

def load_and_cache_examples(data_processor, cached_dataset_root, data_type='train'):
    if data_type == 'train':
        cached_train_dataset_file = cached_dataset_root + r'/' + data_type + r'.pt'  # './data/dataset_cache/cdr/train.pt'
        if not os.path.exists(cached_train_dataset_file):
            train_samples = data_processor.get_train_sample()
            # train_samples为parsed_sent_indices, parsed_sent_segments, parsed_sent_attention_masks, parsed_ner_labels, parsed_nen_labels, parsed_real_lens 6个元素组成的元组,详情见data_process.py中Processor类的__parse_data方法
            torch.save(train_samples, cached_train_dataset_file)
        else:
            train_samples = torch.load(cached_train_dataset_file)
        return NERNENDataset(train_samples, mode=data_type)
    elif data_type == 'eval':
        cached_eval_dataset_file = cached_dataset_root + r'/' + data_type + r'.pt'  # './data/dataset_cache/cdr/eval.pt'
        if not os.path.exists(cached_eval_dataset_file):
            eval_samples = data_processor.get_eval_sample()
            torch.save(eval_samples, cached_eval_dataset_file)
        else:
            eval_samples = torch.load(cached_eval_dataset_file)
        return NERNENDataset(eval_samples, mode=data_type)
    else:
        cached_test_dataset_file = cached_dataset_root + r'/' + data_type + r'.pt'  # './data/dataset_cache/cdr/test.pt'
        if not os.path.exists(cached_test_dataset_file):
            test_samples = data_processor.get_test_sample()
            torch.save(test_samples, cached_test_dataset_file)
        else:
            test_samples = torch.load(cached_test_dataset_file)
        return NERNENDataset(test_samples, mode=data_type)


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    # 魔改collate_fn
    # batch：长为24的列表，每个元素都是长为4的元组

    all_sentences, all_ners, all_nens, all_lens = batch
    max_len = max(all_lens).item()
    all_sentences['input_ids'] = all_sentences['input_ids'][:, :max_len]
    all_sentences['token_type_ids'] = all_sentences['token_type_ids'][:, :max_len]
    all_sentences['attention_mask'] = all_sentences['attention_mask'][:, :max_len]
    all_ners = all_ners[:, :max_len]
    all_nens = all_nens[:, :max_len]
    return [all_sentences, all_ners, all_nens, all_lens]

if __name__ == '__main__':
    cached_train_dataset_file = './data/dataset_cache/train.pt'
    if not os.path.exists(cached_train_dataset_file):
        from data_process import Processor
        DICT_DATASET = {
                    "train": "CDR/train.txt",
                    "dev": "CDR/dev.txt",
                    "test": "CDR/test.txt",
                    "zs_test": "CDR/zs_test.txt",
                    }

        bert_path = '/Users/wangyang/Desktop/wy/project/python/deeplearning/nernen/data/model_data/biobert-base-cased-v1.2' # TODO 测试用，在服务器上部署时要换成biobert-large-cased-v1.1
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(bert_path) # 分词器
        p = Processor(tokenizer, DICT_DATASET)
        train_samples = p.get_train_sample()
        torch.save(train_samples, cached_train_dataset_file)
    else:
        train_samples = torch.load(cached_train_dataset_file)

    train_dataset = NERNENDataset(train_samples, mode='train')
    # train_dataset[0]:
    # (
    #     {
    #     'input_ids': tensor([  101,  9468,  2858, 21501,  1162,  7936,  1116,  1103,  2848,  7889,
    #                             17786,  5026,  2109,  2629,  1104,   172,  4934,  2386,  2042,   119,
    #                             102,     0,     0,     0,     0,     0,     0,     0,     0,     0,
    #                             ..., 
    #                             0,     0,     0,     0,     0,     0,     0,     0,     0,     0]), # torch.Size([128])
    #     'token_type_ids': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                               ...,
    #                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), # torch.Size([128])
    #     'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
    #                               ...,
    #                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # torch.Size([128])
    #     },
    #     tensor([2, 1, 0, 0, 0, 2, 0, 2, 2, 0, 0, 0, 0, 2, 2, 1, 0, 0, 0, 2, 2, 2, 2, 2,
    #             ...,
    #             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]), # torch.Size([128])
    #     tensor([2, 1, 0, 0, 0, 2, 0, 2, 2, 0, 0, 0, 0, 2, 2, 3, 0, 0, 0, 2, 2, 2, 2, 2,
    #             ...,
    #             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]) # torch.Size([128])
    # )
    print(len(train_dataset)) # 5819
    
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    for INPUT, NER, NEN, LEN in train_dataloader:
        print(INPUT['input_ids'].shape)# torch.Size([batch_size, 128])
        print(INPUT['token_type_ids'].shape)# torch.Size([batch_size, 128])
        print(INPUT['attention_mask'].shape)# torch.Size([batch_size, 128])
        print(NER.shape)# torch.Size([batch_size, 128])
        print(NEN.shape)# torch.Size([batch_size, 128])
        print(LEN.shape)# torch.Size([batch_size])
        print(LEN)
        break