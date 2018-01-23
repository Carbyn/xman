#!/usr/bin/python 
import paddle.v2 as paddle
import json
from reader import load_json
from network import lstm_net
import gzip
import cPickle
import math
import numpy as np


def infer(data_type):
    proposal_data = {'results': {}, 'version': "VERSION 1.0"}
    json_data = load_json("data/meta.json")
    database = json_data['database']
    dict_dim = 2048
    class_num = 2 
    window_lens = [30]
    window_stride = 30 
    model_path = "model/video_lstm30.tar.gz"
    prob_layer = lstm_net(dict_dim, class_num, is_infer=True)

    # initialize PaddlePaddle
    paddle.init(use_gpu=False, trainer_count=1)

    # load the trained models
    if os.path.exists(model_path):
        with gzip.open(model_path, 'r') as f:
            parameters = paddle.parameters.Parameters.from_tar(f)
    index = 0
    for video in database.keys():
        dataSet = database[video]["subset"]
        if dataSet != data_type:
            continue
        with open("data/" + dataSet + "/" + str(video) + ".pkl", 'rb') as f:
            video_fea = cPickle.load(f)
        print index, video
        index += 1
        video_len = np.shape(video_fea)[0]
        this_vid_proposals = []
        for pos in range(0, video_len, window_stride):
            inputs = []
            for window_len in window_lens:
                if pos + window_len < video_len:
                    inputs.append([video_fea[pos:pos + window_len]])
            probs = paddle.infer(
                output_layer=prob_layer, parameters=parameters, input=inputs, field="value")
            if len(probs) <= 0:
                continue
            max_probs = np.max(probs[0])
            window_index = 0 
            label = np.argmax(probs[0])
            if label == 0:
                continue
            proposal = {
                    'label': label, 
                    'score': int(max_probs * 100) / 100.0,
                    'segment': [pos, pos + window_lens[window_index]],
                   }
            this_vid_proposals += [proposal]
        proposal_data['results'][video] = this_vid_proposals
    with open("res/" + data_type + "_len.json", 'w') as fobj:
        json.dump(proposal_data, fobj)
#infer('testing')
infer('validation')
