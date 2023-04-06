# -*- coding: utf-8 -*-
# @Time    : 2023/4/4 11:02
# @Author  : huangkai
# @File    : label_tool.py
"""
single-pass

"""
import numpy as np
# import numpy as np
# from scipy.spatial.distance import pdist#直接调包可以计算JC值 :需要两个句子长度一样；所以暂时不用
import jieba


def Jaccrad(model, reference):  # terms_reference为源句子，terms_model为候选句子
    terms_reference = jieba.cut(reference)  # 默认精准模式
    terms_model = jieba.cut(model)
    grams_reference = set(terms_reference)  # 去重；如果不需要就改为list
    grams_model = set(terms_model)
    temp = 0
    for i in grams_reference:
        if i in grams_model:
            temp = temp + 1
    fenmu = len(grams_model) + len(grams_reference) - temp  # 并集
    jaccard_coefficient = float(temp / fenmu)  # 交集
    return jaccard_coefficient


a = "香农在信息论中提出的信息熵定义为自信息的期望"
b = "信息熵作为自信息的期望"
jaccard_coefficient = Jaccrad(a, b)
print(jaccard_coefficient)


def single_pass(corpus, theta):
    """
    2020-6-5 将single_pass聚类结果平均句向量，然后计算置信度，获取置信度
    :param corpus:
    :param theta:
    :return:{topic:["",""],topic:["",""]}
    """
    cluster_text = {}
    num_topic = 0

    # bert_text = dict(zip(corpus, corpus_vec))

    # 否则执行single-pass聚类算法
    for index, text in enumerate(range(len(corpus))):
        if num_topic == 0:
            cluster_text.setdefault(num_topic, []).append(text)
            num_topic += 1
        else:
            result = []
            for i in range(num_topic):
                temp_result = []
                for txt in cluster_text[i]:
                    jaccard_score = Jaccrad(txt, text)
                    temp_result.append(jaccard_score)
                temp_score = np.mean(temp_result)
                result.append(temp_score)
            max_score_idx, max_score = np.argmax(result), np.max(result)

            if max_score > theta:
                cluster_text[int(max_score_idx)].append(text)
            else:
                cluster_text[num_topic + 1] = [text]
                num_topic += 1

    return cluster_text
