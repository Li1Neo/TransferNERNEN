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


数据集中有许多nen标签为-1的样本


def __parse_nen_tag(nen_tag):
    """
    parse the nen tag sequence to the dictionary 将nen标签序列解析为字典
    example:
        nen_tag:['D009270', 'O', 'O', 'O', 'O', 'O', 'D003000', 'O'] -> result:{'D009270': [(0, 1)], 'D003000': [(6, 1)]}
    """
    tmp = []
    result = {}

    for i in range(len(nen_tag)):
        if nen_tag[i] != "O" :
            if(i==0): tmp.append(i)
            else :
                if(nen_tag[i-1]=="O") : tmp.append(i)
                else :
                    if(nen_tag[i]==nen_tag[i-1]) : tmp.append(i)
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
    return result

nen_tag = ['D009270', 'D009270', 'D009080', 'O', 'O', 'D009080', 'D009080', 'O', 'O', 'D003000', 'O', 'D003000', 'D003000']
print(__parse_nen_tag(nen_tag))


train_s_ind: 大小为(6, 100)的tensor  6为batch_size, 100为maxlen
tf.Tensor([[101, 16421, 131, 50352, 32188, 1988, 1110, 1126, 3903, 3252, 1111,  2581, 188, 1181, 10645, 1118, 34574, 4889, 119, 102, 0, 0, 0, ..., 0, 0, 0],# 每行100个元素，共6行
           ...,
           [101, 1103, 3507, 4184, 7889, 23652, 6360, 1104, 8920, 16655, 18885, 12090, 23601, 1197, 11759, 1110, 1136, 3106, 4628, 117, 1133, 2554, 5401, 1115, 6484, 2489, 32169, 20267, 5968, 3053, 1107, 7987, 3242, 119, 102, 0, 0, 0, ..., 0, 0, 0]])

train_s_seg: 大小为(6, 100)的全0tensor

train_e_ind: 大小为(6, 16)的tensor
tf.Tensor([[101, 50352, 32188, 1988, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 每行16个元素，共6行
           ..., 
           [101, 16655, 18885, 12090, 23601, 1197, 4091, 11759, 102, 0, 0, 0, 0, 0, 0, 0]])

train_e_seg: 大小为(6, 16)的全0tensor

train_ner: 大小为(6, 100)的tensor
tf.Tensor([[2, 2, 2, 1, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 0, 2, 2, 2, 2, ... , 2, 2, 2],
           ..., 
           [2, 2, 2, 0, 0, 0, 0, 2, 2, 3, 0, 0, 0, 0, 5, 2, 2, 2, 2, 2, 2, 2,... , 2, 2, 2]])

train_cpt_ner: 大小为(6, 100)的tensor
tf.Tensor([[2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 1, 1, 2, 2, 2, 2, ... , 2, 2, 2],
           ..., 
           [2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 5, 2, 2, 2, 2, 2, 2, 2, ... , 2, 2, 2]])

train_nen: 大小为(6,)的tensor
tf.Tensor([1, 0, 1, 1, 1, 1])

_: 大小为(6,)的tensor
tf.Tensor([14, 8, 47, 17, 17, 24])





one_hot(train_ner, LABEL_SIZE): 大小为(6, 100, 6)的tensor
tf.Tensor([[[0., 0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0.],
            ...,
            [0., 0., 1., 0., 0., 0.]],
           [[0., 0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0.],
            ...,
            [0., 0., 1., 0., 0., 0.]],
           [[0., 0., 1., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0.],
            ...,
            [0., 0., 1., 0., 0., 0.]],
           ...,
           [[0., 0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0.],
            ...,
            [0., 0., 1., 0., 0., 0.]]])


one_hot(train_cpt_ner, LABEL_SIZE): 大小为(6, 100, 6)的tensor
tf.Tensor([[[0., 0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0.],
            ...,
            [0., 0., 1., 0., 0., 0.]],
           [[0., 0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0.],
            ...,
            [0., 0., 1., 0., 0., 0.]],
           [[0., 0., 1., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0.],
            ...,
            [0., 0., 1., 0., 0., 0.]],
           ...,
           [[0., 0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0.],
            ...,
            [0., 0., 1., 0., 0., 0.]]])

one_hot(train_nen, 2): 大小为(6, 2)的tensor
tf.Tensor([[1., 0.],
           [0., 1.],
           [0., 1.],
           [1., 0.],
           [1., 0.],
           [1., 0.]])

logits_ner, logits_cpt_ner, logits_nen = model(train_s_ind, train_s_seg, train_e_ind, train_e_seg)

-tf.reduce_sum(one_hot(train_ner, LABEL_SIZE) * tf.math.log(logits_ner), axis=-1)