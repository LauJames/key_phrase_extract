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
import re
from rake_nltk import Rake


curdir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
sys.path.insert(0, curdir)

json_path = curdir + '/all_title_abstract_keyword_clean.json'
rake_process_path = curdir + '/rake_extract_keyphrase.json'
rake_process_path_12 = curdir + '/rake_extract_keyphrase_12.json'
rake_process_txt_path = curdir + '/rake_extract_keyphrase.txt'
stop_words_path = curdir + '/stopwords.txt'


def load_stop_words(stop_word_file):
    """
    Utility function to load stop words from a file and return as a list of words
    @param stop_word_file Path and file name of a file containing stop words.
    @return list A list of stop words.
    """
    stop_words = []
    for line in open(stop_word_file):
        if line.strip()[0:1] != "#":
            for word in line.split():  # in case more than one per line
                stop_words.append(word)
    return stop_words


rake = Rake(max_length=4, stopwords=load_stop_words(stop_words_path))


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
        json_line = fin.readline()  # just one line
        json_obj = json.loads(json_line)
    return json_obj


def extract_keyphrase(json_obj, save_path):
    with open(save_path, 'w+', encoding='utf8') as fout:
        json_list = []
        for temp in json_obj:
            title = temp["title"]
            abstract = temp["abstract"]
            key_words = temp["keyword"]
            extract_text = title + '. ' + abstract
            extract_text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9~!@#$%^&*()_+<>?:,./;’，。、‘：“《》？~！@#￥%……（）]', ' ',
                                  extract_text)
            rake.extract_keywords_from_text(extract_text)
            rake_string = []
            count = 0
            for tup in rake.get_ranked_phrases_with_scores():
                if count >= 10:
                    break
                rake_string.append(str(tup[0]) + '|||' + tup[1])
                count += 1
            extracted_info = '###'.join(rake_string)
            # extracted_info = ';'.join([str(tup[0]) + '|||' + tup[1] for tup in rake.get_ranked_phrases_with_scores()])
            json_str = {"extract_text": extract_text, "keywords": key_words, "rake_extract": extracted_info}
            json_list.append(json_str)
        json.dump(json_list, fout, ensure_ascii=False)


def extract_keyphrase2txt(json_obj, save_path):
    with open(save_path, 'w+', encoding='utf8') as fout:
        for temp in json_obj:
            title = temp["title"]
            abstract = temp["abstract"]
            key_words = temp["keyword"]
            extract_text = title + '. ' + abstract
            extract_text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9~!@#$%^&*()_+<>?:,./;’，。、‘：“《》？~！@#￥%……（）]', ' ',
                                  extract_text)
            rake.extract_keywords_from_text(extract_text)
            rake_string = []
            count = 0
            for tup in rake.get_ranked_phrases_with_scores():
                if count > 12:
                    break
                rake_string.append(str(tup[0]) + '|||' + tup[1])
                count += 1
            extracted_info = '###'.join(rake_string)
            # extracted_info = ';'.join([str(tup[0]) + '|||' + tup[1] for tup in rake.get_ranked_phrases_with_scores()])
            txt_str = extract_text + '\t' + key_words + '\t' + extracted_info
            fout.write(txt_str + '\n')


if __name__ == '__main__':
    # rake_test()
    json_obj = load_json(json_path)
    extract_keyphrase(json_obj=json_obj, save_path=rake_process_path_12)
    # extract_keyphrase2txt(json_obj=json_obj, save_path=rake_process_txt_path)
    # str_test = "Stable haptic rendering with detailed energy-compensating control ?. Sampled-data system nature is the main factor for a haptic system to exhibit non-passive behaviors or instabilities through energy leaks, especially for stiff objects rendering. A detailed energy-compensating method is presented aiming to improve the haptic system's performance based on the concept of doing work. Using an ideal continuous-time haptic system as a reference, we calculate the work difference between interactions in the sampled-data haptic system and the counterparts in the real world. An energy-compensating controller (ECC) is then designed to compensate for, in every sampling period, the energy leaks caused by work difference. The work difference exists both in entering and leaving periods of interaction contacts, which means the ECC not only removes the unwanted extra work from the virtual environment to eliminate potential non-passive system behaviors, but also compensates for the deficient work that should be done by the human operator to guarantee the intended rendering stiffness. The proposed method was tested and demonstrated on six human subjects with the implementation of a stiff-wall prototype haptic system via a Delta haptic device.	energy-compensating controller;work difference;perceived stiffness;haptic system;passivity theory "
    # str_test = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9~!@#$%^&*()_+<>?:,./;’，。、‘：“《》？~！@#￥%……（）]', ' ', str_test)
    # print(str_test)
