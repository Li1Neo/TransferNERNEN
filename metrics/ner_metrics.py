import torch
from collections import Counter
from utils import get_entities

class SeqEntityScore(object):
    def __init__(self, id2label, markup='bios'):
        self.id2label = id2label # {0: 'X', 1: 'B-CONT', 2: 'B-EDU', 3: 'B-LOC', 4: 'B-NAME', 5: 'B-ORG', 6: 'B-PRO', 7: 'B-RACE', 8: 'B-TITLE', 9: 'I-CONT', 10: 'I-EDU', 11: 'I-LOC', 12: 'I-NAME', 13: 'I-ORG', 14: 'I-PRO', 15: 'I-RACE', 16: 'I-TITLE', 17: 'O', 18: 'S-NAME', 19: 'S-ORG', 20: 'S-RACE', 21: '[START]', 22: '[END]'}
        self.markup = markup # 'bios'
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
        # self.origins：长为1489的列表
        # [['ORG', 2, 3], ['TITLE', 4, 7], ['TITLE', 9, 12], ... , ['TITLE', 55, 58], ['ORG', 0, 2], ['TITLE', 3, 6]]
        origin_counter = Counter([x[0] for x in self.origins])
        # origin_counter：Counter({'TITLE': 686, 'ORG': 522, 'NAME': 109, 'EDU': 105, 'CONT': 32, 'PRO': 18, 'RACE': 15, 'LOC': 2})
        found_counter = Counter([x[0] for x in self.founds])
        right_counter = Counter([x[0] for x in self.rights])
        for type_, count in origin_counter.items(): # 如type_：'ORG', count：522
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
        # {'acc': precision, 'recall': recall, 'f1': f1} 为总体的f1、P、R
        # class_info为各个类别的f1、P、R：
        # {
        #     'ORG': {'acc': ..., 'recall': ..., 'f1': ...},
        #     'TITLE': {'acc': ..., 'recall': ..., 'f1': ...},
        #     'NAME': {'acc': ..., 'recall': ..., 'f1': ...},
        #     'RACE': {'acc': ..., 'recall': ..., 'f1': ...},
        #     'EDU': {'acc': ..., 'recall': ..., 'f1': ...},
        #     'CONT': {'acc': ..., 'recall': ..., 'f1': ...},
        #     'PRO': {'acc': ..., 'recall': ..., 'f1': ...},
        #     'LOC': {'acc': ..., 'recall': ..., 'f1': ...}
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
            label_entities = get_entities(label_path, self.id2label, self.markup)
            # [['ORG', 2, 3], ['TITLE', 4, 7], ['TITLE', 9, 12]]
            pre_entities = get_entities(pre_path, self.id2label, self.markup)
            # [['RACE', 1, 1], ['RACE', 3, 3], ['RACE', 4, 4], ['RACE', 5, 5], ['RACE', 6, 6], ['RACE', 7, 7], ['RACE', 9, 9], ['RACE', 10, 10], ['RACE', 11, 11]]
            self.origins.extend(label_entities) # [['ORG', 2, 3], ['TITLE', 4, 7], ['TITLE', 9, 12]] # 真实实体
            self.founds.extend(pre_entities) # [['RACE', 1, 1], ['RACE', 3, 3], ['RACE', 4, 4], ['RACE', 5, 5], ['RACE', 6, 6], ['RACE', 7, 7], ['RACE', 9, 9], ['RACE', 10, 10], ['RACE', 11, 11]] # 预测实体
            self.rights.extend([pre_entity for pre_entity in pre_entities if pre_entity in label_entities]) # 预测正确的实体

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



