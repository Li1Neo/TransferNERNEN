import torch
from collections import Counter
from utils import get_entities

class SeqEntityScore(object):
    def __init__(self, id2label, markup='bios'):
        self.id2label = id2label # {0: 'X', 1: 'B-Chemical', 2: 'O', 3: 'B-Disease', 4: 'I-Chemical', 5: 'I-Disease'}
        self.markup = markup # 'bio'
        self.reset()

    def reset(self):
        self.origins = [] # 所有真实实体
        self.founds = []  # 所有识别的实体
        self.rights = []  # 所有识别正确的实体

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        # self.origins：长为1489的列表
        # [['Disease', 0, 8], ['Chemical', 10, 13], ['Disease', 14, 14], ... , ['Chemical', 10, 13], ['Chemical', 1, 4], ['Chemical', 11, 14]]
        origin_counter = Counter([x[0] for x in self.origins])
        # origin_counter：Counter({'Chemical': 5321, 'Disease': 4182})
        found_counter = Counter([x[0] for x in self.founds])
        # found_counter：Counter({'Chemical': 5301, 'Disease': 4118})
        right_counter = Counter([x[0] for x in self.rights])
        # right_counter：Counter({'Chemical': 4868, 'Disease': 3270})
        for type_, count in origin_counter.items(): # 如type_：'Disease', count：4182
            origin = count # 4182
            found = found_counter.get(type_, 0) # 4118
            right = right_counter.get(type_, 0) # 3270
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)} # class_info['Disease'] = {'acc': 0.7941, 'recall': 0.7819, 'f1': 0.788}
        origin = len(self.origins) # 9503
        found = len(self.founds) # 9419
        right = len(self.rights) # 8138
        recall, precision, f1 = self.compute(origin, found, right) # 0.8563611491108071, 0.8639983013058711, 0.8601627734911742
        return {'acc': precision, 'recall': recall, 'f1': f1}, class_info
        # {'acc': 0.8639983013058711, 'recall': 0.8563611491108071, 'f1': 0.8601627734911742} 为总体的P、R、f1
        # class_info为各个类别的P、R、f1：
        # {
        #     'Disease': {'acc': 0.7941, 'recall': 0.7819, 'f1': 0.788},
        #     'Chemical': {'acc': 0.9183, 'recall': 0.9149, 'f1': 0.9166}
        # }

    def update(self, label_paths, pred_paths):
        '''
        labels_paths: [[],[],[],....]
        pred_paths: [[],[],[],.....]

        :param label_paths:
        :param pred_paths:
        :return:
        Example:
            >>> labels_paths = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
            >>> pred_paths = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        '''
        for label_path, pre_path in zip(label_paths, pred_paths):
            # pre_path: [3, 5, 5, 5, 5, 5, 5, 5, 5, 2, 1, 4, 4, 4, 3, 2, 2, 2, 2, 2]
            # label_path: ['B-Disease', 'I-Disease', 'I-Disease', 'I-Disease', 'I-Disease', 'I-Disease', 'I-Disease', 'I-Disease', 'I-Disease', 'O', 'B-Chemical', 'I-Chemical', 'I-Chemical', 'I-Chemical', 'B-Disease', 'O', 'O', 'O', 'O', 'O']
            label_entities = get_entities(label_path, self.id2label, self.markup)
            # [['Disease', 0, 8], ['Chemical', 10, 13], ['Disease', 14, 14]] 一句话中的真实实体
            pre_entities = get_entities(pre_path, self.id2label, self.markup)
            # [['Disease', 0, 8], ['Chemical', 10, 13], ['Disease', 14, 14]] 一句话中的预测实体
            self.origins.extend(label_entities) # [['Disease', 0, 8], ['Chemical', 10, 13], ['Disease', 14, 14]] # 将一句话中的真实实体添加到self.origins
            self.founds.extend(pre_entities) # [['Disease', 0, 8], ['Chemical', 10, 13], ['Disease', 14, 14]] # 将一句话中的预测实体添加到self.founds
            self.rights.extend([pre_entity for pre_entity in pre_entities if pre_entity in label_entities]) # [['Disease', 0, 8], ['Chemical', 10, 13], ['Disease', 14, 14]]
            # 将一句话中预测正确的实体添加到self.rights

class SpanEntityScore(object):
    def __init__(self, id2label):
        self.id2label = id2label
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        origin_counter = Counter([self.id2label[x[0]] for x in self.origins])
        found_counter = Counter([self.id2label[x[0]] for x in self.founds])
        right_counter = Counter([self.id2label[x[0]] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'acc': precision, 'recall': recall, 'f1': f1}, class_info

    def update(self, true_subject, pred_subject):
        self.origins.extend(true_subject)
        self.founds.extend(pred_subject)
        self.rights.extend([pre_entity for pre_entity in pred_subject if pre_entity in true_subject])



