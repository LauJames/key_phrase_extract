#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : key_phrase_extract 
@File     : dataHelper.py
@Time     : 2019/3/5 10:44
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""

import os
import sys
import json
from rake_nltk import Rake

rake = Rake()

curdir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
sys.path.insert(0, curdir)

json_path = curdir + '/all_title_abstract_keyword_clean.json'


def rake_test():
    my_test = 'My father was a self-taught mandolin player. He was one of the best string instrument players ' \
              'in our town. He could not read music, but if he heard a tune a few times, he could play it. ' \
              'When he was younger, he was a member of a small country music band. They would play at local dances ' \
              'and on a few occasions would play for the local radio station. He often told us how he had auditioned ' \
              'and earned a position in a band that featured Patsy Cline as their lead singer. He told the family that ' \
              'after he was hired he never went back. Dad was a very religious man. He stated that there was a lot of ' \
              'drinking and cursing the day of his audition and he did not want to be around that type of environment.'
    rake.extract_keywords_from_text(my_test)
    print(rake.get_ranked_phrases())
    print(rake.get_ranked_phrases_with_scores())
    print(rake.get_word_degrees())


def load_json(path):
    with open(path, 'r', encoding='utf8') as fin:
        json_line = fin.readlines()
        json_obj = json.loads(json_line)
    return json_obj


