#! /user/bin/evn python
# -*- coding:utf8 -*-

import codecs
import numpy as np
import nltk

# 获取每篇文档的topK个融合的关键术语
def get_topK_kp(all_merged_kp, k):
    topK_merged_kp = []
    for i in range(len(all_merged_kp)):
        sorted_list = sorted(all_merged_kp[i].items(), key=lambda d: d[1], reverse=True)
        one_doc_kp_list = []
        for j in range(k):
            one_doc_kp_list.append(sorted_list[j][0])
        topK_merged_kp.append(one_doc_kp_list)
    return topK_merged_kp


def evaluate(topK_merged_kp, original_kp):
    precision = []
    recall = []
    # k可能小于标准关键术语个数
    doc_num = len(topK_merged_kp)
    for i in range(doc_num):
        #  计算每一篇文档的p和r
        correct_num = 0
        for j in range(len(topK_merged_kp[i])):
            if original_kp[i].__contains__(topK_merged_kp[i][j]):
                correct_num += 1
        pi = correct_num / len(topK_merged_kp[i])
        ri = correct_num / len(original_kp[i])
        precision.append(pi)
        recall.append(ri)
    # 计算全部文档的平均p和r
    precision = np.array(precision)
    recall = np.array(recall)
    precision_avg = np.average(precision)
    recall_avg = np.average(recall)
    f = (2 * precision_avg * recall_avg) / (precision_avg + recall_avg)

    return precision_avg, recall_avg, f, precision, recall


# 去停用词和词干提取
def stemming(kp_list, stop_words):
    stemmer = nltk.stem.PorterStemmer()
    all_stem_result = []
    for i in range(len(kp_list)):
        one_stem_result = []
        for j in range(len(kp_list[i])):
            one_kp_split = kp_list[i][j].split(' ')
            one_stem_kp = stemmer.stem(one_kp_split[0])
            for k in range(1, len(one_kp_split)):
                if not stop_words.__contains__(one_kp_split[k]):
                    one_stem_kp = one_stem_kp + ' ' + stemmer.stem(one_kp_split[k])
            one_stem_result.append(one_stem_kp)
        all_stem_result.append(one_stem_result)
    return all_stem_result


def evaluate_stem(topK_merged_kp, original_kp, stop_words):
    topK_merged_kp = stemming(topK_merged_kp, stop_words)
    original_kp = stemming(original_kp, stop_words)
    precision = []
    recall = []
    # k可能小于标准关键术语个数
    doc_num = len(topK_merged_kp)
    for i in range(doc_num):
        # print('关键术语topK: ' + str(topK_merged_kp[i]))
        # print('原始关键术语：' + str(original_kp[i]))
        #  计算每一篇文档的p和r
        correct_num = 0
        for j in range(len(topK_merged_kp[i])):
            if original_kp[i].__contains__(topK_merged_kp[i][j]):
                correct_num += 1
        pi = correct_num / len(topK_merged_kp[i])
        ri = correct_num / len(original_kp[i])
        precision.append(pi)
        recall.append(ri)
    # 计算全部文档的平均p和r
    precision = np.array(precision)
    recall = np.array(recall)
    precision_avg = np.average(precision)
    recall_avg = np.average(recall)
    f = (2 * precision_avg * recall_avg) / (precision_avg + recall_avg)

    return precision_avg, recall_avg, f, precision, recall


def save_results(result_array, save_path):
    # fp = open(file=save_dir, mode='w', encoding='utf-8')
    fp = codecs.open(filename=save_path, mode='w', encoding='utf-8')
    for i in range(len(result_array)):
        line = str(result_array[i])
        fp.write(str(i) + ":" + line + '\n')
    fp.close()


# 对融合的全部结果排序后写入文件
def save_all_merged_results(result_list, save_dir):
    fp = codecs.open(filename=save_dir, mode='w', encoding='utf-8')
    for i in range(len(result_list)):
        line = str(sorted(result_list[i].items(), key=lambda d: d[1], reverse=True))
        fp.write(line + '\n')
    fp.close()

