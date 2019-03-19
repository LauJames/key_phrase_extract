#python example to infer document vectors from trained doc2vec model
import gensim.models as g
import codecs

#parameters
model="toy_data/model.bin"
test_docs="toy_data/test_docs.txt"
output_file="toy_data/test_vectors.txt"

#inference hyper-parameters
start_alpha=0.01
infer_epoch=1000

#load model
m = g.Doc2Vec.load(model)
test_docs = [x.strip().split() for x in codecs.open(test_docs, "r", "utf-8").readlines()]

# Gensim 中有内置的 most_similar，得到向量后，可以计算相似性
# 返回 topn个相关的doc和similarity
doc = 'virtually enhancing the perception of user actions. This paper proposes using virtual reality to enhance the' \
      ' perception of actions by distant users on a shared application. Here, distance may refer either to space ' \
      '( e.g. in a remote synchronous collaboration) or time ( e.g. during playback of recorded actions). Our ' \
      'approach consists in immersing the application in a virtual inhabited 3D space and mimicking user actions ' \
      'by animating avatars. We illustrate this approach with two applications, the one for remote collaboration ' \
      'on a shared application and the other to playback recorded sequences of user actions. We suggest this could ' \
      'be a low cost enhancement for telepresence.'
doc1 = 'modeling 3d facial expressions using geometry videos. The significant advances in developing high speed ' \
       'shape acquisition devices make it possible to capture the moving and deforming objects at video speeds.' \
       ' However, due to its complicated nature, it is technically challenging to effectively model and store ' \
       'the captured motion data. In this paper, we present a set of algorithms to construct geometry videos' \
       ' for 3D facial expressions, including hole filling, geodesic based face segmentation, and expression' \
       ' invariant parametrization. Our algorithms are efficient and robust, and can guarantee the exact ' \
       'correspondence of the salient features (eyes, mouth and nose). Geometry video naturally bridges the ' \
       '3D motion data and 2D video, and provides a way to borrow the well studied video processing techniques' \
       ' to motion data processing. With our proposed intra frame prediction scheme based on H.264/AVC, we are' \
       ' able to compress the geometry videos into a very compact size while maintaining the video quality.' \
       ' Our experimental results on real world datasets demonstrate that geometry video is effective for ' \
       'modeling the high resolution 3D expression data.'

docs = []
vector_list = []
docs.append(doc)
docs.append(doc1)
# vector= " ".join([str(x) for x in m.infer_vector(doc, alpha=start_alpha, steps=infer_epoch)])
# for doc in docs:
#     vector = " ".join([str(x) for x in m.infer_vector(docs, alpha=start_alpha, steps=infer_epoch)])
#     vector_list.append(vector)
# print(vector)

# inferred_vector_dm = model_dm.infer_vector(test_text.split(' '))
# # print inferred_vector_dm
# # Gensim 中有内置的 most_similar，得到向量后，可以计算相似性
# sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=10)
print(m.docvecs[0])
inferred_vector_dm = m.infer_vector(doc.split(' '))
# sims = m.docvecs.most_similar([m.docvecs[0]], topn=3)
sims = m.docvecs.most_similar([inferred_vector_dm], topn=3)
for sim in sims:
    print(sim[1])



#infer test vectors
# output = open(output_file, "w")
# for d in test_docs:
#     output.write(" ".join([str(x) for x in m.infer_vector(d, alpha=start_alpha, steps=infer_epoch)]) + "\n")
# output.flush()
# output.close()
