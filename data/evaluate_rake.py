#! /user/bin/evn python
# -*- coding:utf8 -*-
import os
import re
import numpy as np
from numpy import linalg
import time, datetime
import gensim
import gensim.models as g
from ir.config import Config
from ir.search import Search
from data import data_IO, evaluate

if __name__ == '__main__':
    file_path = 'doc_test.txt'
    file_path_json = 'rake_extract_keyphrase_12.json'
    # evaluate dir：
    evaluate_dir = '../evaluate_rake/'
    if not os.path.exists(evaluate_dir):
        os.makedirs(evaluate_dir)
    data_num = 100000
    p_list = [0.2, 0.5, 0.6, 0.8]
    k_list = [2, 4, 6, 8, 10, 12]
    # p_list = [0.2]
    # k_list = [2]
    stop_words = data_IO.get_stopword()


    # prepare for data
    ids, _, all_doc_keywords,all_rake_dict = data_IO.load_all_data_json4(file_path_json)  #全量

    all_doc_keywords = all_doc_keywords[0:100000]
    all_rake_dict = all_rake_dict[0:100000]
    print('abstract_str_list.len: ' + str(len(all_doc_keywords)))


    # merge:
    start_time = time.time()
    avg_evaluate = {}
    for k in k_list:
        print('取前 ' + str(k) + ' 个关键术语的结果：')
        # 取topK个关键词：
        topK_merged_kp = evaluate.get_topK_kp(all_rake_dict, k)

        # evaluate: 结果stemming后进行评估
        precision_avg, recall_avg, f, precision, recall = evaluate.evaluate_stem(topK_merged_kp, all_doc_keywords,
                                                                                     stop_words)
        print('平均检准率： ', precision_avg)
        print('平均检全率： ', recall_avg)
        print('F值： ', f)

        print('\n')
        avg_evaluate.update({k: (precision_avg, recall_avg, f)})

    avg_dir = ('../evaluate_rake_10w/evaluate_rake_avg12.txt')
    print(avg_dir)
    with open(avg_dir, mode='w', encoding='utf-8')as wp:
        for i in avg_evaluate:
            wp.write('k='+str(i) + ': ' + str(avg_evaluate.get(i)) + '\n')
        print('评估结果存储完毕！')


    end_time = time.time()
    time_used = datetime.timedelta(seconds=int(round(end_time - start_time)))
    print('评估总体耗时： ', str(time_used))






