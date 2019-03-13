#! /user/bin/evn python
# -*- coding:utf8 -*-


from elasticsearch import Elasticsearch


class Config(object):
    def __init__(self):
        print("config...")
        self.es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
        self.index_name = "key_phrase_data_index"
        self.doc_type = "all_data"

        file_path = '../data/rake_extract_keyphrase.json'
        self.doc_path = file_path


def main():
    Config()


if __name__ == '__main__':
    main()
