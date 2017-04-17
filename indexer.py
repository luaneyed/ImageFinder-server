import os
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

import elasticsearch
from elasticsearch import helpers

INDEX = 'step2_index'
modelPath = os.getcwd() + '/model'
predictionNum = 5

if len(sys.argv) < 2:
    print('\nPlease pass a relative path of directory as a argument.\n')
    sys.exit()

es = elasticsearch.Elasticsearch("localhost:9200")

print('If you want to preserve image index, append -p argument')
if not (len(sys.argv) > 2 and sys.argv[2] == '-p'):
    es.indices.delete(index = INDEX, ignore = [400, 404])

if es.indices.exists(INDEX) == False:
    es.indices.create(
        index = INDEX,
        body = {
            'settings': {
                'index': {
                    'number_of_shards': 1,
                    'number_of_replicas': 1
                },
                'analysis': {
                    'analyzer': {
                        'descriptor_analyzer': {
                            'tokenizer': 'array_tokenizer'
                        }
                    },
                    'tokenizer': {
                        'array_tokenizer': {
                            'type': 'pattern',
                            'pattern': '/'
                        }
                    }
                }
            },
            'mappings': {
                'hash_value': {
                    'properties': {
                        'path': {
                            'type': 'string'
                        },
                        'predictions': {
                            'type': 'string',
                            'analyzer': 'descriptor_analyzer'
                        },
                        'scores': {
                            'type': 'string'
                        }
                    }
                }
            }
        }
    )



with tf.gfile.FastGFile(os.path.join(modelPath, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name = '')


relativePath = sys.argv[1] if len(sys.argv) > 1 else '/'
if relativePath[0] != '/':
    relativePath = '/' + relativePath
if relativePath[-1] == '/':
    relativePath = relativePath[0:-1]

docs = []
rootDir = os.getcwd() + relativePath
with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')

    for dirName, subdirList, fileList in os.walk(rootDir):
        print('Indexing directory: %s' % dirName)
        for fname in fileList:
            path = dirName + '/' + fname
            try:
                size = os.path.getsize(path)
            except:
                continue
            extention = os.path.splitext(path)[1]
            if (size > 0) and (extention == '.jpg' or extention == '.jpeg' or extention == '.png') :
                print('path : %s' % path)
                image_data = tf.gfile.FastGFile(path, 'rb').read()
                predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
                predictions = np.squeeze(predictions)
                topPrediction = predictions.argsort()[-predictionNum:][::-1]

                predictionsToIndex = []
                scoresToIndex = []
                for prediction in topPrediction:
                    predictionsToIndex.append(str(prediction))
                    scoresToIndex.append("{:.5f}".format(predictions[prediction]))

                docs.append({
                    '_index' : INDEX,
                    '_type' : "hash_value",
                    '_id': path,
                    '_source': {
                        'path': path,
                        'predictions': '/'.join(predictionsToIndex),
                        'scores': '/'.join(scoresToIndex)
                    }
                })
                if len(docs) >= 1000:
                    helpers.bulk(es, docs)
                    docs = []
