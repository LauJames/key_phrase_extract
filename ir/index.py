#! /user/bin/evn python
# -*- coding:utf8 -*-


from ir.config import Config
from elasticsearch import helpers
import json


class Index(object):
    def __init__(self):
        print("Indexing ...")

    @staticmethod
    def data_convert(file_path="../data/rake_extract_keyphrase.json"):
        print("convert json file into single doc")

        all_data = {}
        data_count = 0
        file = open(file_path, encoding='utf-8')
        json_dict = json.load(file)
        for one_doc in json_dict:
            keywords = one_doc['keywords']
            abstract = one_doc['extract_text']
            rake_extract = one_doc['rake_extract']
            all_data[data_count] = {'keywords': keywords, 'abstract': abstract, 'rake_extract':rake_extract}
            data_count += 1

        return all_data

    @staticmethod
    def create_index(config):
        print("creating %s index ..."%config.index_name)
        request_body = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "similarity": {
                    "LM": {
                        "type": "LMJelinekMercer",
                        "lambda": 0.4
                    }
                }
            },
            "mapping": {
                config.index_name: {
                    "properties": {
                        "id": {
                            "type": "long",
                            "index": False
                        },
                        "keywords": {
                            "type": "text",
                            "term_vector": "with_positions_offsets_payloads",
                            "store": True,
                            "analyzer": "standard",
                            "similarity": "LM"
                        },
                        "abstract": {
                            "type": "text",
                            "term_vector": "with_positions_offsets_payloads",
                            "store": True,
                            "analyzer": "standard",
                            "similarity": "LM"
                        },
                        "rake_extract": {
                            "type": "text",
                            "term_vector": "with_positions_offsets_payloads",
                            "store": True,
                            "analyzer": "standard",
                            "similarity": "LM"
                        }
                    }
                }
            }
        }

        config.es.indices.delete(index=config.index_name, ignore=[400, 404])
        res = config.es.indices.create(index=config.index_name, body=request_body)
        print(res)
        print("Indices are created successfully")

    @staticmethod
    def bulk_index(all_data, bulk_size, config):
        print("Bulk index for question")
        count =1
        actions = []
        for data_count, data in all_data.items():
            action = {
                "_index": config.index_name,
                "_type": config.doc_type,
                "_id": data_count,
                "_source": data
            }
            actions.append(action)
            count += 1

            if len(actions) % bulk_size == 0:
                helpers.bulk(config.es, actions)
                print("Bulk index: " + str(count))
                actions = []

        if len(actions) > 0:
            helpers.bulk(config.es, actions)
            print("Bulk index: " + str(count))


def main():
    config = Config()
    index = Index()
    all_data = index.data_convert(config.doc_path)
    index.create_index(config)
    index.bulk_index(all_data, bulk_size=10000, config=config)


if __name__ == '__main__':
    main()
