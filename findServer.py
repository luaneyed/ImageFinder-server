import os
import sys
import json
import base64
import elasticsearch
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request
from flask_cors import CORS, cross_origin

INDEX = 'step2_index'
modelPath = os.getcwd() + '/model'
predictionNum = 5

app = Flask(__name__)
CORS(app)

es_client = elasticsearch.Elasticsearch("localhost:9200")


def calcScore(predictions1, predicsions2, scores1, scores2):
    if scores1 == scores2:
        return float("{:.5f}".format(5.0))
    score = 0
    for i in range(predictionNum):
        for j in range(predictionNum):
            if predictions1[i] == predicsions2[j]:
                score = score + float(scores1[i]) * float(scores2[j])
                break
    return float("{:.5f}".format(score))

with tf.gfile.FastGFile(os.path.join(modelPath, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name = '')

def appendCandidates(candidates, docs, predictionsToSearch, scoresToSearch):
    for doc in docs.get('hits').get('hits'):
        source = doc.get('_source')
        score = calcScore(predictionsToSearch, source.get('predictions').split('/'),
            scoresToSearch, source.get('scores').split('/'))
        if score > 0:
            candidates.append({
                'path': source.get('path'),
                'score': score
            })
    return candidates

def predictImage(image_data):
    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')

        # image_data = tf.gfile.FastGFile(path, 'rb').read()
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)
        topPrediction = predictions.argsort()[-predictionNum:][::-1]

        predictionsToSearch = []
        scoresToSearch = []
        for prediction in topPrediction:
            predictionsToSearch.append(str(prediction))
            scoresToSearch.append("{:.5f}".format(predictions[prediction]))

        candidates = []
        docs = es_client.search(
            index = INDEX,
            doc_type = 'hash_value',
            body = {
                'query': {
                    'terms': {
                        'predictions': predictionsToSearch
                    }
                }
            },
            scroll = '1m',
            size = 1000
        )
        sid = docs.get('_scroll_id')
        rest = docs.get('hits').get('total')
        candidates = appendCandidates(candidates, docs, predictionsToSearch, scoresToSearch)

        while rest > 0:
            docs = es_client.scroll(scroll_id = sid, scroll = '1m')
            sid = docs.get('_scroll_id')
            rest = len(docs.get('hits').get('hits'))
            candidates = appendCandidates(candidates, docs, predictionsToSearch, scoresToSearch)

        candidates = sorted(candidates, key = (lambda c: c.get('score')), reverse = True)
        resultNum = len(candidates)
        bestScore = candidates[0].get('score')
        hasAnswer = bestScore == 5.0
        for i in range(len(candidates)):
            candidate = candidates[i]
            score = candidate.get('score')
            if (hasAnswer and (score < 0.7)) or ((i >= 500) and (score < bestScore / 2)) or \
                ((score < bestScore / 4 ) and (score < 0.5)):
                resultNum = i
                break

        return candidates[:resultNum]

queryImageName = ''
results = []
resultsNum = 0

def makeGETresponse(results, newNext):
    newResults = []
    for result in results:
        path = result.get('path')
        with open(path, "rb") as imageFile:
          f = imageFile.read()  #   bytes
          b = base64.b64encode(f).decode('utf-8')
        newResults.append({
            'score': result.get('score'),
            'path': result.get('path'),
            'image': b
        })
    return {
        'results': newResults,
        'next': newNext
    }

@app.route('/')
def hello_luan():
    return 'Hello Luan!'

@app.route('/find', methods = ['GET', 'POST'])
def handleRequest():
    global results
    global resultsNum
    global queryImageName

    if request.method == 'POST':
        image = dict(request.files).get('image')[0].stream.read()
        results = predictImage(image)
        resultsNum = len(results)
        queryImageName = dict(request.form).get('name')[0]
        return json.dumps(results)
    else:
        imageName = dict(request.args).get('imageName')[0]
        receivedNext = dict(request.args).get('next')[0]
        if receivedNext == '':
            receivedNext = 0
        else:
            receivedNext = int(receivedNext)
        if imageName != queryImageName:
            return json.dumps({'results': []})
        if receivedNext >= resultsNum:
            return json.dumps({'results': []})
        if receivedNext + 50 >= resultsNum:
            return json.dumps(makeGETresponse(results[receivedNext:], None))
        return json.dumps(makeGETresponse(
            results[receivedNext:(receivedNext + 50)], receivedNext + 50
            ))

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 8080)




