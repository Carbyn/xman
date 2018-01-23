import paddle.v2 as paddle
import json
import argparse
from reader import load_json
from network import lstm_net
import gzip
import cPickle
import math
import numpy as np


def infer(video, st, ed):
    proposal_data = {'results': {}, 'version': "VERSION 1.0"}
    json_data = load_json("data/meta.json")
    database = json_data['database']
    dict_dim = 2048
    class_num = 2 
    dataSet = 'validation'
    model_path = "model/video_lstm30.tar.gz"
    prob_layer = lstm_net(dict_dim, class_num, is_infer=True)

    # initialize PaddlePaddle
    paddle.init(use_gpu=False, trainer_count=1)

    # load the trained models
    if os.path.exists(model_path):
        with gzip.open(model_path, 'r') as f:
            parameters = paddle.parameters.Parameters.from_tar(f)
    with open("data/" + dataSet + "/" + str(video) + ".pkl", 'rb') as f:
        video_fea = cPickle.load(f)
    probs = paddle.infer(
        output_layer=prob_layer, parameters=parameters, input=[[video_fea[int(st):int(ed)]]], field="value")
    print probs


def parse_input():
    p = argparse.ArgumentParser()
    p.add_argument('video')
    p.add_argument('st')
    p.add_argument('ed')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_input()
    infer(**vars(args))
