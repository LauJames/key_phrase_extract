#! /user/bin/evn python
# -*- coding:utf8 -*-

from data import data_IO
import codecs

docs_text, _  = data_IO.load_json_data_for_es('rake_extract_keyphrase.json')
with codecs.open('./toy_data/train_docs.txt',  mode='w', encoding='utf-8') as fw:
    for text in docs_text:
        fw.write(text + '\n')
    print('文本数据提取完毕！')
