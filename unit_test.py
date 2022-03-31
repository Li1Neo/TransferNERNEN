#!python
# coding=utf-8
'''
:Description  : To be filled
:Version      : 1.0
:Author       : LilNeo
:Date         : 2022-03-16 19:26:38
:LastEditors  : wy
:LastEditTime : 2022-03-19 00:06:09
:FilePath     : /nernen/unit_test.py
:Copyright 2022 LilNeo, All Rights Reserved. 
'''


# 数据集中有许多nen标签为-1的样本

import copy
def __parse_nen_tag(nen_tag, id2label):
    """
    parse the nen tag sequence to the dictionary 将nen标签序列解析为字典
    example:
        # nen_tag:['D009270', 'O', 'O', 'O', 'O', 'O', 'D003000', 'O']
        # ->
        # result:{'D009270': [(0, 1)], 'D003000': [(6, 1)]}

        nen_tag:['D009270', 'D009270', 'D009080', 'O', 'O', 'D009080', 'D009080', 'O', 'O', 'D003000', 'O', 'D003000', 'D003000']
        ->
        [['D009270', 0, 1], ['D009080', 2, 2], ['D009080', 5, 6], ['D003000', 9, 9], ['D003000', 11, 12]]
    """
    tmp = []
    result = {}
    if not isinstance(nen_tag[0], str):
        nen_tag = [id2label[tag] for tag in nen_tag]
    for i in range(len(nen_tag)):
        if nen_tag[i] != "O":
            if i == 0: tmp.append(i)
            else:
                if nen_tag[i-1] == "O": tmp.append(i)
                else:
                    if nen_tag[i] == nen_tag[i-1]: tmp.append(i)
                    else:
                        if len(tmp):
                            cache = result.get(nen_tag[i-1], [])
                            cache.append((tmp[0], len(tmp)))
                            result[nen_tag[i - 1]] = cache
                            tmp.clear()
                            tmp.append(i)
        else:
            if len(tmp):
                cache = result.get(nen_tag[i-1], [])
                cache.append((tmp[0], len(tmp)))
                result[nen_tag[i - 1]] = cache
                tmp.clear()
    if len(tmp):
        i = len(nen_tag)
        cache = result.get(nen_tag[i-1], [])
        cache.append((tmp[0], len(tmp)))
        result[nen_tag[i - 1]] = cache
        tmp.clear()
    # result: {'D009270': [(0, 2)], 'D009080': [(2, 1), (5, 2)], 'D003000': [(9, 1), (11, 2)]}
    chunks = []
    chunk = [-1, -1, -1]
    for key, value in result.items():
        # 如key：'D009080'
        # value：[(2, 1), (5, 2)]
        for idx in value:
            chunk[0] = key
            chunk[1] = idx[0]
            chunk[2] = idx[0] + idx[1] - 1
            chunks.append(copy.deepcopy(chunk))
    return chunks



nen_tag = ['D009270', 'D009270', 'D009080', 'O', 'O', 'D009080', 'D009080', 'O', 'O', 'D003000', 'O', 'D003000', 'D003000']
print(__parse_nen_tag(nen_tag, ''))
# [['D009270', 0, 1], ['D009080', 2, 2], ['D009080', 5, 6], ['D003000', 9, 9], ['D003000', 11, 12]]