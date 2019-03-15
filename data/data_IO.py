#! /user/bin/evn python
# -*- coding:utf8 -*-

import re
import codecs
import numpy as np
import json



def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    :param string:
    :return: string
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    # string = re.sub('[_A-Z ]', ' ', string)
    return string.strip().lower()


def get_stopword():
    stop_words = []
    with codecs.open(filename='stopwords.txt', encoding='utf-8') as fp:
        while True:
            line = fp.readline().strip()
            if not line:
                print('停用词加载完毕！')
                return stop_words
            stop_words.append(line)


# 加载词典
def load_vocab(vacab_dir):
    vocab = []
    with codecs.open(filename=vacab_dir, encoding='utf-8') as fp:
        while True:
            line = fp.readline()
            if not line:
                print('get vacab successful!')
                return vocab

            tmp = line.strip().split(' ')
            vocab.append(tmp[0])


# txt
def load_all_data(txt_file_path, vocab):
    print('开始读取txt数据...')
    docs = []
    key_phrases = []
    key_phrase_extracs = []
    line_num = 0

    with codecs.open(filename=txt_file_path, encoding='utf-8') as fp:
        while True:
            is_error = False
            line_num += 1
            print(line_num)
            line = fp.readline()
            if not line:
                print('txt数据读取完毕!')
                # docs = [clean_str(str(doc)) for doc in docs]
                return [docs, key_phrases, key_phrase_extracs]

            tmp = line.strip().split('\t')

            extracs_tmp = tmp[2].split(';')
            doc_phrase_weight = {}
            for i in range(len(extracs_tmp)):
                extracs_phrase_weight = extracs_tmp[i].split('|||')
                try:
                    doc_phrase_weight.update({extracs_phrase_weight[1]: float(extracs_phrase_weight[0])})
                except (Exception) as e:
                    print('Exception：' + str(e))
                    print('该行提取的关键术语数据有误：' + str(tmp[2]))
                    print('具体数据错误：' + str(extracs_phrase_weight))
                    i = len(extracs_tmp) + 1
                    is_error = True
                    continue

            if not is_error:
                key_phrase_extracs.append(doc_phrase_weight)
                doc_split = clean_str(tmp[0]).split(' ')
                for m in range(len(doc_split)):
                    if not vocab.__contains__(doc_split[m]):
                        doc_split[m] = 'unknown'
                docs.append(doc_split)

                key_phrases.append(tmp[1].split(';'))

            # 按value升序排序
            # doc_phrase_weight = sorted(doc_phrase_weight.items(), key=operator.itemgetter(1))

            # 按value值降序排序 =================转成了list
            # doc_phrase_weight = sorted(doc_phrase_weight.items(), key=lambda d: d[1], reverse=True)


            # print(key_phrase_extracs[:,:,0:3] )
            # print(doc_phrase_weight[0:3] +  '   '+ str(doc_phrase_weight[0][1]))

            # print(tmp[2])
            # print("=====" + str(doc_phrase_weight) + '\n')


# json
def load_all_data_json(json_file_path, vocab):
    print('开始读取json数据...')
    docs = []
    key_phrases = []
    key_phrase_extracs = []

    file = open(json_file_path, encoding='utf-8')
    json_dict = json.load(file)
    for one_doc in json_dict:
        is_error = False
        keywords = one_doc['keywords']
        doc_text = one_doc['extract_text']
        rake_extract = one_doc['rake_extract']

        extracs_tmp = rake_extract.split('###')
        doc_phrase_weight = {}
        # ============================================
        # 添加判断 如果出现异常 舍弃整条数据
        for i in range(len(extracs_tmp)):
            extracs_phrase_weight = extracs_tmp[i].split('|||')
            try:
                doc_phrase_weight.update({extracs_phrase_weight[1]: float(extracs_phrase_weight[0])})
            except (Exception) as e:
                print('Exception:', str(e))
                print('该行提取的关键术语数据有误：' + str(rake_extract))
                print('具体数据错误：' + str(extracs_phrase_weight))
                i = len(extracs_tmp) + 1
                is_error = True
                continue

        if not is_error:
            # 添加抽取的关键词
            key_phrase_extracs.append(doc_phrase_weight)
            # 添加摘要文本
            doc_split = clean_str(doc_text).split(' ')
            for m in range(len(doc_split)):
                if not vocab.__contains__(doc_split[m]):
                    doc_split[m] = 'unknown'
            docs.append(doc_split)
            # 添加原始关键术语
            key_phrases.append(keywords.split(';'))

    print('json数据读取完毕!')
    return [docs, key_phrases, key_phrase_extracs]


# return 未切分的abstract 和 切分的keywords
def load_all_data_for_es(txt_file_path):
    print('开始读取txt数据...')
    abstracts_str = []
    key_phrases = []
    line_num = 0

    with codecs.open(filename=txt_file_path, encoding='utf-8') as fp:
        while True:
            is_error = False
            line_num += 1
            print(line_num)
            line = fp.readline()
            if not line:
                print('txt数据读取完毕!')
                # docs = [clean_str(str(doc)) for doc in docs]
                return [abstracts_str, key_phrases]

            tmp = line.strip().split('\t')

            extracs_tmp = tmp[2].split(';')
            doc_phrase_weight = {}
            for i in range(len(extracs_tmp)):
                extracs_phrase_weight = extracs_tmp[i].split('|||')
                try:
                    doc_phrase_weight.update({extracs_phrase_weight[1]: float(extracs_phrase_weight[0])})
                except (Exception) as e:
                    print('Exception：' + str(e))
                    print('该行提取的关键术语数据有误：' + str(tmp[2]))
                    print('具体数据错误：' + str(extracs_phrase_weight))
                    i = len(extracs_tmp) + 1
                    is_error = True
                    continue

            if not is_error:
                abstracts_str.append(tmp[0])
                key_phrases.append(tmp[1].split(';'))

            # 按value升序排序
            # doc_phrase_weight = sorted(doc_phrase_weight.items(), key=operator.itemgetter(1))

            # 按value值降序排序 =================转成了list
            # doc_phrase_weight = sorted(doc_phrase_weight.items(), key=lambda d: d[1], reverse=True)


            # print(key_phrase_extracs[:,:,0:3] )
            # print(doc_phrase_weight[0:3] +  '   '+ str(doc_phrase_weight[0][1]))

            # print(tmp[2])
            # print("=====" + str(doc_phrase_weight) + '\n')


# return 未切分的abstract 和 切分的keywords
def load_json_data_for_es(json_file_path):
    print('开始读取json数据...')
    abstracts_str = []
    key_phrases = []
    # key_phrase_extracs = []

    file = open(json_file_path, encoding='utf-8')
    json_dict = json.load(file)
    for one_doc in json_dict:
        is_error = False
        keywords = one_doc['keywords']
        doc_text = one_doc['extract_text']
        rake_extract = one_doc['rake_extract']

        extracs_tmp = rake_extract.split('###')
        doc_phrase_weight = {}
        # ============================================
        # 添加判断 如果出现异常 舍弃整条数据
        for i in range(len(extracs_tmp)):
            extracs_phrase_weight = extracs_tmp[i].split('|||')
            try:
                doc_phrase_weight.update({extracs_phrase_weight[1]: float(extracs_phrase_weight[0])})
            except (Exception) as e:
                print('Exception:', str(e))
                print('该行提取的关键术语数据有误：' + str(rake_extract))
                print('具体数据错误：' + str(extracs_phrase_weight))
                i = len(extracs_tmp) + 1
                is_error = True
                continue

        if not is_error:
            # 添加抽取的关键词
            # key_phrase_extracs.append(doc_phrase_weight)
            # 添加摘要文本
            abstracts_str.append(doc_text)
            # 添加原始关键术语
            key_phrases.append(keywords.split(';'))

    print('json数据读取完毕!')
    return [abstracts_str, key_phrases]

# 获取文章的abstract
def get_abstracts_str(file_path):
    docs = []
    with codecs.open(filename=file_path, encoding='utf-8') as fp:
        while True:
            line = fp.readline()
            if not line:
                print('get docs successful!')
                return docs

            tmp = line.strip().split('\t')
            docs.append(tmp[0])
            # doc_split = tmp[0].split(' ')
            # for m in range(len(doc_split)):
            #     if not vocab.__contains__(doc_split[m]):
            #         doc_split[m] = 'unknown'
            # docs.append(doc_split)




# 加载词向量/计算文档向量
def doc2vec(vector_model, docs):
    all_doc_vectors = []
    # v1 = word2vec_model['virtually']
    # print(v1)
    # word2vev_model = gensim.models.keyedvectors._load_word2vec_format(datapath(vector_dir), binary=False)
    for i in range(len(docs)):
        doc = docs[i]
        # print(doc)
        vector = np.zeros(300)
        for j in range(len(doc)):
            vector = vector + np.array(vector_model[doc[j]])
        # 文档向量：词向量取均值
        doc_vector = vector / len(doc)
        all_doc_vectors.append(doc_vector)
    print('文档向量计算完毕！\n')
    return np.array(all_doc_vectors)


# 对内部抽取和外部融合后的dict的weight做归一化 （weight - min） /(max - min)
def normalization(kp_weight_dict):
    max_weight = max(kp_weight_dict.values())
    min_weight = min(kp_weight_dict.values())
    scale = max_weight - min_weight
    for key in kp_weight_dict:
        weight = (kp_weight_dict[key] - min_weight) / scale
        kp_weight_dict.update({key: weight})
    return kp_weight_dict


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


if __name__ == '__main__':
    vector_dir = 'sg.word2vec.300d'
    file_path = 'doc_test.txt'
    file_path_json = 'rake_extract_keyphrase.json'
    vocab_dir = 'vocab_sg300d.txt'
    merged_results_dir = 'all_merged_results.txt'
    # evaluate dir：
    evaluate_dir = '../evaluate/'
    topK_merged_dir = 'topK_merged_results.txt'
    precision_dir = 'precision.txt'
    recall_dir = 'recall.txt'