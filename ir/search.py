#! /user/bin/evn python
# -*- coding:utf8 -*-


from ir.config import Config


class Search(object):
    def __init__(self):
        print("Searching ...")

    @staticmethod
    def search_by_abstract(abstract, top_n, config):
        q = {
            "query": {
                "multi_match": {
                    "query": abstract,
                    "fields": ["abstract"],
                    "fuzziness": "AUTO"
                }
            }
        }

        count = 0
        while count < top_n:
            try:
                res = config.es.search(index=config.index_name, doc_type=config.doc_type, body=q, request_timeout=30)
                topn = res['hits']['hits']
                count = 0
                result = []
                for data in topn:
                    if count < top_n:
                        result.append((data['_id'], data['_source']['abstract'], data['_source']['keywords'],
                                       data['_source']['rake_extract']))
                        count += 1
                return result
            except Exception as e:
                print(e)
                print("Try again!")
                count += 1
                continue

        print("ReadTimeOutError may not be covered!")
        return []


def main():
    config = Config()
    search = Search()
    # abstract = "virtually enhancing the perception of user actions. This paper proposes using virtual reality to " \
    #            "enhance the perception of actions by distant users on a shared application. Here, distance may refer " \
    #            "either to space ( e.g. in a remote synchronous collaboration) or time ( e.g. during playback of " \
    #            "recorded actions). Our approach consists in immersing the application in a virtual inhabited 3D space " \
    #            "and mimicking user actions by animating avatars. We illustrate this approach with two applications, " \
    #            "the one for remote collaboration on a shared application and the other to playback recorded sequences " \
    #            "of user actions. We suggest this could be a low cost enhancement for telepresence."
    abstract = 'modeling 3d facial expressions using geometry videos. The significant advances in developing high ' \
               'speed shape acquisition devices make it possible to capture the moving and deforming objects at video' \
               ' speeds. However, due to its complicated nature, it is technically challenging to effectively model' \
               ' and store the captured motion data. In this paper, we present a set of algorithms to construct ' \
               'geometry videos for 3D facial expressions, including hole filling, geodesic based face segmentation, ' \
               'and expression invariant parametrization. Our algorithms are efficient and robust, and can guarantee ' \
               'the exact correspondence of the salient features (eyes, mouth and nose). Geometry video naturally ' \
               'bridges the 3D motion data and 2D video, and provides a way to borrow the well studied video' \
               ' processing techniques to motion data processing. With our proposed intra frame prediction scheme ' \
               'based on H.264/AVC, we are able to compress the geometry videos into a very compact size while' \
               ' maintaining the video quality. Our experimental results on real world datasets demonstrate that ' \
               'geometry video is effective for modeling the high resolution 3D expression data.'
    result = search.search_by_abstract(abstract, 5, config)
    for data in result:
        print(data[0], data[1], data[2])
        # print(data[:2])
    # print(result)


if __name__ == '__main__':
    main()
