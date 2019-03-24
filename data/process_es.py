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

# 使用es获取指定文档数量的topn篇相关文档（相似性计算文档）
def get_es_results(abstracts, top_n):
    start_time = time.time()
    es_results = []
    config = Config()
    search = Search()
    for abstract in abstracts:
        try:
            result = search.search_by_abstract(abstract, top_n, config)
            print(result)
            print('搜索结果中包含 ' + str(len(result)) + ' 条数据')
            es_results.append(result)
        except (Exception) as e:
            print('ES检索出现异常： Exception:', str(e))

    end_time = time.time()
    time_used = datetime.timedelta(seconds= int (round(end_time - start_time)))
    print('检索耗时:' + str(time_used))
    return es_results

# 计算一篇文档与es结果集中文档的相似度 并按相似度降序排序:
# [(1,1),(2,0.8),(3,0.5)...],[(1,0.6),(2,1),(3,0.5)...]
def calculate_doc_sim(doc_vectors):
    # v1:目标文档（ES结果集中的第一条）
    v1_sim = {}
    v1 = doc_vectors[0]
    for i in range(len(doc_vectors)):
        v2 = doc_vectors[i]
        v1_v2_dot = np.dot(v1, v2)
        denom = linalg.norm(v1) * linalg.norm(v2)
        cos = v1_v2_dot / denom  # 余弦值
        v1_sim.update({i: cos})

    # 按value值降序排序 ============v1_sim转换成了list
    v1_sim = sorted(v1_sim.items(), key=lambda d: d[1], reverse=True)

    # print('文档相似度计算完毕！\n')
    return v1_sim


# 对于一篇文档： 融合其topN篇相似的外部文档的全部key phrase
def get_external(topN_doc_sims, keywords, currunt_docID):
    # topN_doc_sims:[(6,0.9),(10,0.8),(3,0.5)...],topN * 2 一篇文档的相似文档及相似度集合
    # keywords: es结果集中的所有文档的原始关键术语
    external_key_phrase = {}
    key_phrases = {}  # {'k1':[0.2,0.3]; 'k2':[0.5,0.6,0.8]}
    for sim_doc in topN_doc_sims:
        # 获取第i篇与本篇doc相似的文档id
        sim_docID = sim_doc[0]
        # 获取第i篇与本篇doc相似的文档sim
        sim = sim_doc[1]

        # 跳过当前文档
        if sim_docID != currunt_docID:
            # 根据相似文档id获取相似文档的关键术语
            sim_doc_keys = keywords[sim_docID]
            for key in sim_doc_keys:
                if not key_phrases.__contains__(key):
                    key_phrases.update({key: [sim]})
                else:
                    sim_list = key_phrases[key]
                    sim_list.append(sim)
                    key_phrases.update({key: sim_list})

    #  计算每个key phrase的权重均值
    for key in key_phrases:
        sim_array = np.array(key_phrases[key])
        # 融合权重：取均值
        # key_weight = np.average(sim_array)
        # 融合权重：求和
        key_weight = np.sum(sim_array)
        external_key_phrase.update({key: key_weight})
    return external_key_phrase


# 对于一篇文档： 融合其topN篇相似的外部文档的全部key phrase
def get_external_doc2vec(topN_doc_sims, keywords, currunt_docID):
    # topN_doc_sims:[(6,0.9),(10,0.8),(3,0.5)...],topN * 2 一篇文档的相似文档及相似度集合
    # keywords: es结果集中的所有文档的原始关键术语
    external_key_phrase = {}
    key_phrases = {}  # {'k1':[0.2,0.3]; 'k2':[0.5,0.6,0.8]}
    for i in range(len(topN_doc_sims)):
        # 获取第i篇与本篇doc相似的文档的sim值
        sim = topN_doc_sims[i][1]

        # 跳过当前文档
        if i != currunt_docID:
            # 根据相似文档id获取相似文档的关键术语
            sim_doc_keys = keywords[i]
            for key in sim_doc_keys:
                if not key_phrases.__contains__(key):
                    key_phrases.update({key: [sim]})
                else:
                    sim_list = key_phrases[key]
                    sim_list.append(sim)
                    key_phrases.update({key: sim_list})

    #  计算每个key phrase的权重均值
    for key in key_phrases:
        sim_array = np.array(key_phrases[key])
        # 融合权重：取均值
        # key_weight = np.average(sim_array)
        # 融合权重：求和
        key_weight = np.sum(sim_array)
        external_key_phrase.update({key: key_weight})
    return external_key_phrase

# 对于一篇文档：融合内外部关键术语
# 目标文档本身权重 p  外部文档权重 1-p
def merge(original_dict, external_dict, p):
    merge_dict = {}
    # all_keys = original_dict.keys() | external_dict.keys()
    for original_key in original_dict:
        # 原文档有 外部文档没有
        if not external_dict.__contains__(original_key):
            weight = p * original_dict[original_key]
        # 原文档有 外部文档也有
        else:
            weight = p * original_dict[original_key] + (1 - p) * external_dict[original_key]
        merge_dict.update({original_key: weight})

    # 原文档没有 外部文档有
    for external_key in external_dict:
        if not merge_dict.__contains__(external_key):
            weight = (1 - p) * external_dict[external_key]
            merge_dict.update({external_key: weight})

    return merge_dict


# 对一篇文档：
# def extract_es(es_result, vector_model, vocab, topN, p):
#     print('提取当前文档的关键词：')
#     start_time = time.time()
#     all_merged_kp = []

# 对所有文档：
def extract_all_es(es_results, vector_model, vocab, topN, p):
    print('extract_all_es:...')
    start_time = time.time()
    all_merged_kp = []
    # 对一篇文档：
    for es_result in es_results:
        # es_result 包含目标文档的数据
        is_error = False
        # 获取当前文档的rake抽取结果
        rake_extract = es_result[0][3]  # 目标文档在es 搜索结果的第一条
        # 处理目标文档的rake_extract
        rake_extract_dict = {}
        extracs_tmp = rake_extract.split('###')
        for m in range(len(extracs_tmp)):
            extracs_phrase_weight = extracs_tmp[m].split('|||')
            try:
                rake_extract_dict.update({extracs_phrase_weight[1]: float(extracs_phrase_weight[0])})
            except (Exception) as e:
                print('Exception:', str(e))
                print('该行提取的关键术语数据有误：' + str(rake_extract))
                print('具体数据错误：' + str(extracs_phrase_weight))
                is_error = True
                m = len(extracs_tmp) + 1
                continue
        if not is_error:
            abstracts = []
            keywords = []
            for data in es_result:
                # 获取当前文档的es检索结果文档
                abs_split = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9~!@#$%^&*()_+<>?:,./;’，。、‘：“《》？~！@#￥%……（）]', ' ', data[1]).split(' ')
                for j in range(len(abs_split)):
                    if not vocab.__contains__(abs_split[j]):
                        abs_split[j] = 'unknown'
                abstracts.append(abs_split)

                # 获取结果文档的原始关键术语
                keywords.append(data[2].split(';'))

            doc_vectors = data_IO.doc2vec(vector_model, abstracts)
            doc_sims = calculate_doc_sim(doc_vectors)
            # 根据向量相似度大小取topN篇相似文档
            topN_doc_sims = doc_sims[:topN + 1]  # 相似文档里包含里目标文档本身

            external_dict = get_external(topN_doc_sims, keywords, currunt_docID=0)

            # 添加归一化操作
            external_dict = data_IO.normalization(external_dict)
            rake_extract_dict = data_IO.normalization(rake_extract_dict)

            one_merge_dict = merge(rake_extract_dict, external_dict, p)
            all_merged_kp.append(one_merge_dict)
    end_time = time.time()
    time_used = datetime.timedelta(seconds=int(round(end_time - start_time)))
    print('耗时： ', str(time_used))
    return all_merged_kp



# 计算全部文档的rake_dict和external_dict
def get_all_merge_info(es_results, vector_model, vocab, topN):
    print('get_all_merge_info:...')
    start_time = time.time()
    all_merged_info= []
    # 对一篇文档：
    for es_result in es_results:
        # es_result 包含目标文档的数据
        is_error = False
        # 获取当前文档的rake抽取结果
        try:
            rake_extract = es_result[0][3]  # 目标文档在es 搜索结果的第一条

        # 处理目标文档的rake_extract
            rake_extract_dict = {}
            extracs_tmp = rake_extract.split('###')
        except (Exception) as e:
            print(e)
            print(es_result)
        for m in range(len(extracs_tmp)):
            extracs_phrase_weight = extracs_tmp[m].split('|||')
            try:
                rake_extract_dict.update({extracs_phrase_weight[1]: float(extracs_phrase_weight[0])})
            except (Exception) as e:
                print('Exception:', str(e))
                print('该行提取的关键术语数据有误：' + str(rake_extract))
                print('具体数据错误：' + str(extracs_phrase_weight))
                is_error = True
                m = len(extracs_tmp) + 1
                continue
        if not is_error:
            abstracts = []
            keywords = []
            for data in es_result:
                # 获取当前文档的es检索结果文档
                abs_split = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9~!@#$%^&*()_+<>?:,./;’，。、‘：“《》？~！@#￥%……（）]', ' ', data[1]).split(' ')
                for j in range(len(abs_split)):
                    if not vocab.__contains__(abs_split[j]):
                        abs_split[j] = 'unknown'
                abstracts.append(abs_split)

                # 获取结果文档的原始关键术语
                keywords.append(data[2].split(';'))

            doc_vectors = data_IO.doc2vec(vector_model, abstracts)
            doc_sims = calculate_doc_sim(doc_vectors)
            # 根据向量相似度大小取topN篇相似文档
            topN_doc_sims = doc_sims[:topN + 1]  # 相似文档里包含里目标文档本身

            external_dict = get_external(topN_doc_sims, keywords, currunt_docID=0)

            # 添加归一化操作
            external_dict = data_IO.normalization(external_dict)
            rake_extract_dict = data_IO.normalization(rake_extract_dict)
            all_merged_info.append([external_dict, rake_extract_dict])

    end_time = time.time()
    time_used = datetime.timedelta(seconds=int(round(end_time - start_time)))
    print('get_all_merge_info()耗时： ', str(time_used))
    return all_merged_info


def get_all_merge_info_doc2vec(es_results, doc2vec_model, topN):
    print('get_all_merge_info_doc2vec:...')
    start_time = time.time()
    all_merged_info = []
    # 对一篇文档：
    for es_result in es_results:
        # es_result 包含目标文档的数据
        is_error = False
        # 获取当前文档的rake抽取结果
        rake_extract = es_result[0][3]  # 目标文档在es 搜索结果的第一条
        # 处理目标文档的rake_extract
        rake_extract_dict = {}
        extracs_tmp = rake_extract.split('###')
        for m in range(len(extracs_tmp)):
            extracs_phrase_weight = extracs_tmp[m].split('|||')
            try:
                rake_extract_dict.update({extracs_phrase_weight[1]: float(extracs_phrase_weight[0])})
            except (Exception) as e:
                print('Exception:', str(e))
                print('该行提取的关键术语数据有误：' + str(rake_extract))
                print('具体数据错误：' + str(extracs_phrase_weight))
                is_error = True
                m = len(extracs_tmp) + 1
                continue
        if not is_error:
            # 获取结果文档的原始关键术语
            keywords = [data[2].split(';') for data in es_result]

            # 使用gensim doc_vecter_model
            doc_id = int(es_result[0][0])
            doc_vector = doc2vec_model.docvecs[doc_id]
            topN_doc_sims = doc2vec_model.docvecs.most_similar([doc_vector], topn=topN)

            external_dict = get_external_doc2vec(topN_doc_sims, keywords, currunt_docID=0)

            # 添加归一化操作
            external_dict = data_IO.normalization(external_dict)
            rake_extract_dict = data_IO.normalization(rake_extract_dict)
            all_merged_info.append([external_dict, rake_extract_dict])

    end_time = time.time()
    time_used = datetime.timedelta(seconds=int(round(end_time - start_time)))
    print('get_all_merge_info()耗时： ', str(time_used))
    return all_merged_info


# 基于每篇文档的rake提取关键词和原始关键词进行内外部关键词的融合
def extract_all(all_merged_info, p):
    start_time = time.time()
    all_merged_kp = []
    for merged_info in all_merged_info:
        external_dict = merged_info[0]
        rake_extract_dict = merged_info[1]
        one_merge_dict = merge(rake_extract_dict, external_dict, p)
        all_merged_kp.append(one_merge_dict)

    end_time = time.time()
    time_used = datetime.timedelta(seconds=int(round(end_time - start_time)))
    print('extract_all()耗时： ', str(time_used))
    return all_merged_kp


if __name__ == '__main__':
    doc2vec_dir = '../doc2vec/model.bin'
    vector_dir = 'sg.word2vec.300d'
    file_path = 'doc_test.txt'
    file_path_json = 'rake_extract_keyphrase.json'
    vocab_dir = 'vocab_sg300d.txt'
    merged_results_dir = 'all_merged_results.txt'
    # es_dir = 'process_es_search.txt'
    # evaluate dir：
    evaluate_dir = '../evaluate_es_doc10w/'
    topK_merged_dir = 'topK_merged_results.txt'
    # precision_dir = 'precision.txt'
    # recall_dir = 'recall.txt'
    # avg_dir = 'avg.txt'
    data_num = 100000
    topN = 10  # 10篇相似文档
    p_list = [0.2, 0.5, 0.6, 0.8]
    k_list = [2, 4, 6]
    # p_list = [0.2]
    # k_list = [2]
    stop_words = data_IO.get_stopword()
    print('加载词向量模型...')
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(fname=vector_dir, binary=False)
    print('词向量模型加载完毕！')
    # print('加载文档向量模型...')
    # doc2vec_model = g.Doc2Vec.load(doc2vec_dir)
    # print('文档向量模型加载完毕！')


    # prepare for data
    vocab = data_IO.load_vocab(vocab_dir)
    abstract_str_list, all_doc_keywords = data_IO.load_json_data_for_es(file_path_json, data_num=data_num)  #读取10w条数据
    print('abstract_str_list.len: ' + str(len(abstract_str_list)))
    # abstract_str_list = data_IO.get_abstracts_str(file_path)
    # es_results = get_es_results(abstract_str_list, top_n=50)
    # data_IO.save_es_search_results(es_results,es_dir)
    es_results = data_IO.load_all_temp_info('es_search.txt')
    all_merged_info = get_all_merge_info(es_results, word2vec_model, vocab, topN)
    data_IO.save_es_search_results(all_merged_info, '../merge_info/10w_merge_info_avg.txt')
    # all_merged_info = get_all_merge_info_doc2vec(es_results, doc2vec_model, topN)
    print(all_merged_info)
    print('计算merge需要的信息完毕！')


    # merge:
    start_time = time.time()
    avg_evaluate = {}

    for p in p_list:
        print('概率p为 ' + str(p) + ' 的结果：')
        if not os.path.exists(evaluate_dir):
            os.makedirs(evaluate_dir)
        p_evaluate_dir = os.path.join(evaluate_dir, 'P' + str(p) + '/')
        if not os.path.exists(p_evaluate_dir):
            os.makedirs(p_evaluate_dir)

        # 以参数p融合内外部关键词
        all_merged_kp = extract_all(all_merged_info, p)

        # print('内外部融合结果：')
        # for i in range(len(all_merged_kp)):
        #     print(sorted(all_merged_kp[i].items(), key=lambda d: d[1], reverse=True))
        all_merged_dir = os.path.join(p_evaluate_dir, 'all_merged.txt')
        evaluate.save_all_merged_results(all_merged_kp, all_merged_dir)

        k_avg_evaluate = []
        for k in k_list:
            print('取前 ' + str(k) + ' 个关键术语的结果：')
            # 文件夹k
            p_k_evaluate_dir = os.path.join(p_evaluate_dir, 'top' + str(k) + '/')
            if not os.path.exists(p_k_evaluate_dir):
                os.makedirs(p_k_evaluate_dir)

            # 取topK个关键词：
            topK_merged_kp = evaluate.get_topK_kp(all_merged_kp, k)
            p_k_merged_results_dir = os.path.join(p_k_evaluate_dir, 'top' + str(k) + '_phrases.txt')
            evaluate.save_results(topK_merged_kp, p_k_merged_results_dir)

            # evaluate: 结果stemming后进行评估
            precision_avg, recall_avg, f, precision, recall = evaluate.evaluate_stem(topK_merged_kp, all_doc_keywords,
                                                                                     stop_words)

            precision_dir = os.path.join(p_k_evaluate_dir, 'precision_' + str(k) + '.txt')
            recall_dir = os.path.join(p_k_evaluate_dir, 'recall_' + str(k) + '.txt')
            evaluate.save_results(precision, precision_dir)
            evaluate.save_results(recall, recall_dir)

            k_avg_evaluate.append({k: [precision_avg, recall_avg, f]})
            print('平均检准率： ', precision_avg)
            print('平均检全率： ', recall_avg)
            print('F值： ', f)

        print('\n')
        avg_evaluate.update({p: k_avg_evaluate})

    avg_dir = os.path.join(evaluate_dir, 'eval_avg_es100.txt')
    print(avg_dir)
    with open(avg_dir, mode='w', encoding='utf-8')as wp:
        for i in avg_evaluate:
            wp.write('p='+str(i) + ': ' + str(avg_evaluate.get(i)) + '\n')
        print('评估结果存储完毕！')


    end_time = time.time()
    time_used = datetime.timedelta(seconds=int(round(end_time - start_time)))
    print('评估总体耗时： ', str(time_used))


    # # merge:
    # start_time = time.time()
    # avg_evaluate = {}
    # for p in p_list:
    #     print('概率p为 ' + str(p) + ' 的结果：')
    #     if not os.path.exists(evaluate_dir):
    #         os.makedirs(evaluate_dir)
    #     p_evaluate_dir = os.path.join(evaluate_dir, 'P' + str(p) + '/')
    #     if not os.path.exists(p_evaluate_dir):
    #         os.makedirs(p_evaluate_dir)
    #
    #     all_merged_dir = os.path.join(p_evaluate_dir, 'all_merged.txt')
    #     all_merged_kp =extract_all_es(es_results, word2vec_model, vocab, topN, p)
    #     # print('内外部融合结果：')
    #     # for i in range(len(all_merged_kp)):
    #     #     print(sorted(all_merged_kp[i].items(), key=lambda d: d[1], reverse=True))
    #     evaluate.save_all_merged_results(all_merged_kp, all_merged_dir)
    #
    #     for k in k_list:
    #         k_avg_evaluate = {}
    #         print('取前 ' + str(k) + ' 个关键术语的结果：')
    #         # 文件夹k
    #         p_k_evaluate_dir = os.path.join(p_evaluate_dir, 'top' + str(k) + '/')
    #         if not os.path.exists(p_k_evaluate_dir):
    #             os.makedirs(p_k_evaluate_dir)
    #
    #         p_k_merged_results_dir = os.path.join(p_k_evaluate_dir, 'top' + str(k) + '_phrases.txt')
    #         topK_merged_kp = evaluate.get_topK_kp(all_merged_kp, k)
    #         evaluate.save_results(topK_merged_kp, p_k_merged_results_dir)
    #
    #         # evaluate:
    #         precision_dir = os.path.join(p_k_evaluate_dir, 'precision_' + str(k) + '.txt')
    #         recall_dir = os.path.join(p_k_evaluate_dir, 'recall_' + str(k) + '.txt')
    #         precision_avg, recall_avg, f, precision, recall = evaluate.evaluate_stem(topK_merged_kp, all_doc_keywords, stop_words)
    #         evaluate.save_results(precision, precision_dir)
    #         evaluate.save_results(recall, recall_dir)
    #
    #         k_avg_evaluate.update({k: [precision_avg, recall_avg, f]})
    #         print('平均检准率： ', precision_avg)
    #         print('平均检全率： ', recall_avg)
    #         print('F值： ', f)
    #
    #     avg_evaluate.update({p: k_avg_evaluate})
    #     print('\n')
    # evaluate.save_results(avg_evaluate, avg_dir)
    #
    # end_time = time.time()
    # time_used = datetime.timedelta(seconds=int(round(end_time - start_time)))
    # print('评估耗时： ', str(time_used))




