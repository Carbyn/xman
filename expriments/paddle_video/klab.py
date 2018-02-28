import cPickle
import sys
import json
import pandas as pd
import numpy as np
import gzip
import os
import random
import paddle.v2 as paddle
import math


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def getLabelSet(data_type):
    data_dict = {}
    json_data = load_json("/mnt/BROAD-datasets/video/meta.json")
    database = json_data['database']
    for video_name in database.keys():
        video_info = database[video_name]
        video_subset = video_info["subset"]
        if video_subset == data_type:
            for index, item in enumerate(video_info['annotations']):
                video_info['annotations'][index] = item.values()[0]
            data_dict[video_name] = video_info['annotations']
    return data_dict


def getLabel(video, pos_st, pos_ed, data_dict):
    max_inter = 0
    for st, ed in data_dict[video]:
        intersection = max(0, min(ed, pos_ed) - max(st, pos_st))
        union = min(max(ed, pos_ed) - min(st, pos_st), ed - st + pos_ed - pos_st)
        overlap = float(intersection) / (union + 1e-8)
        max_inter = max(max_inter, intersection) 
        if overlap > 0.8:
            return 1
    if int(random.random()*100) > 10:
        return -1 
    return 0

def reader_creator(data_type, window_len=0, class_num=2):

    def multi_window_reader():
        json_data = load_json("/mnt/BROAD-datasets/video/meta.json")
        database = json_data['database']
        data_dict = getLabelSet(data_type)
        for video in database.keys():
            dataSet = database[video]["subset"]
            if dataSet != data_type:
                continue
            if int(random.random()*100) > 20:
                continue
            try:
                with open("/mnt/BROAD-datasets/video/" + dataSet + "/image_resnet50_feature/" + str(video) + ".pkl", 'rb') as f:
                    image_fea = np.array(cPickle.load(f))
                with open("/mnt/BROAD-datasets/video/" + dataSet + "/audio_feature/" + str(video) + ".pkl", 'rb') as f:
                    audio_fea = np.array(cPickle.load(f))
            except:
                continue
            image_len = np.shape(image_fea)[0]
            audio_len = np.shape(audio_fea)[0]
            if image_len < audio_len:
                audio_fea = audio_fea[:image_len]
            if audio_len < image_len:
                image_fea = image_fea[:audio_len]
            video_fea = np.append(image_fea, audio_fea, axis=1)
            for wl in window_len:
                for pos in range(0, (np.shape(video_fea)[0] - wl), int(wl * 0.15)):
                    label = getLabel(video, pos, pos + wl, data_dict) 
                    if label < 0:
                        continue
                    yield video_fea[pos:pos + wl], label 
    return multi_window_reader

def lstm_net(input_dim,
             class_dim=2,
             hid_dim=512,
             stacked_num=5,
             is_infer=False):

    assert stacked_num % 2 == 1

    fc_para_attr = paddle.attr.Param(learning_rate=1e-3)
    lstm_para_attr = paddle.attr.Param(initial_std=0., learning_rate=1.)
    para_attr = [fc_para_attr, lstm_para_attr]
    bias_attr = paddle.attr.Param(initial_std=0., l2_rate=0.)
    relu = paddle.activation.Relu()
    linear = paddle.activation.Linear()

    data = paddle.layer.data("video",
                             paddle.data_type.dense_vector_sequence(input_dim))

    fc1 = paddle.layer.fc(input=data,
                          size=hid_dim,
                          act=linear,
                          bias_attr=bias_attr)
    lstm1 = paddle.layer.lstmemory(
        input=fc1, act=relu, bias_attr=bias_attr)

    inputs = [fc1, lstm1]
    for i in range(2, stacked_num + 1):
        fc = paddle.layer.fc(input=inputs,
                             size=hid_dim,
                             act=linear,
                             param_attr=para_attr,
                             bias_attr=bias_attr)
        lstm = paddle.layer.lstmemory(
            input=fc,
            reverse=(i % 2) == 0,
            act=relu,
            bias_attr=bias_attr)
        inputs = [fc, lstm]

    fc_last = paddle.layer.pooling(input=inputs[0], pooling_type=paddle.pooling.Max())
    lstm_last = paddle.layer.pooling(input=inputs[1], pooling_type=paddle.pooling.Max())
    output = paddle.layer.fc(input=[fc_last, lstm_last],
                             size=class_dim,
                             act=paddle.activation.Softmax(),
                             bias_attr=bias_attr,
                             param_attr=para_attr)

    if not is_infer:
        lbl = paddle.layer.data("label", paddle.data_type.integer_value(2))
        cost = paddle.layer.classification_cost(input=output, label=lbl)
        return cost, output, lbl
    else:
        return output
def _train(now_wl, cost, prob, label):
    num_passes = 5 
    window_len = [now_wl]
    model_path = "/home/kesci/work/lstm_av_"  + str(window_len[0])+ ".tar.gz"

    if os.path.exists(model_path):
        with gzip.open(model_path, 'r') as f:
            parameters = paddle.parameters.Parameters.from_tar(f)
    else:
        parameters = paddle.parameters.create(cost)
    adam_optimizer = paddle.optimizer.Adam(                                                                           
        learning_rate=1e-3,
        regularization=paddle.optimizer.L2Regularization(rate=1e-3),                                                  
        model_average=paddle.optimizer.ModelAverage(average_window=0.5))                                              

    # create trainer
    trainer = paddle.trainer.SGD(
        cost=cost,                                                                                                    
        extra_layers=paddle.evaluator.auc(input=prob, label=label),                                                   
        parameters=parameters,                                                                                        
        update_equation=adam_optimizer)                                                                               

    # begin training network                                                                                          
    feeding = {"video": 0, "label": 1}                                                                                 
    def _event_handler(event):
        """
        Define end batch and end pass event handler                                                                   
        """
        if isinstance(event, paddle.event.EndIteration):                                                              
            if event.batch_id % 10 == 0: 
                print "Pass %d, Batch %d, Cost %f, %s\n" % (                                                    
                    event.pass_id, event.batch_id, event.cost, event.metrics)

        if isinstance(event, paddle.event.EndPass): 
            print 'model save'
            with gzip.open(model_path, "w") as f:                                                           
                parameters.to_tar(f)  
            print 'start test'
            result = trainer.test(reader=paddle.batch(reader_creator('validation',window_len,class_num=2), 256), feeding=feeding)                                            
            print "Test at Pass %d, %s \n" % (event.pass_id,                                                
                                                        result.metrics)                                              
                                                                    

    trainer.train(
        reader=paddle.batch(paddle.reader.shuffle(reader_creator('training',window_len,class_num=2), 1280), 256),
        event_handler=_event_handler,
        feeding=feeding,
        num_passes=num_passes)

def train():
    dict_dim = 4096
    paddle.init(use_gpu=True, trainer_count=1)
    # network config                                                                                                  
    cost, prob, label =lstm_net(dict_dim)
    window_lens = [60, 80, 100, 120, 150, 180, 240]
    for wl in window_lens:
        _train(wl, cost ,prob, label)

def infer(data_type):
    prob_layer = lstm_net(4096, is_infer=True)
    paddle.init(use_gpu=True, trainer_count=1)
    def _infer(data_type, window_len, proposal_data, window_stride):
        print window_len
        json_data = load_json("/mnt/BROAD-datasets/video/meta.json")
        database = json_data['database']
        window_lens = [window_len]
        model_path = "/home/kesci/work/lstm_av_" + str(window_len) +".tar.gz"
        # load the trained models
        if os.path.exists(model_path):
            with gzip.open(model_path, 'r') as f:
                parameters = paddle.parameters.Parameters.from_tar(f)
        index = 0
        for video in database.keys():
            dataSet = database[video]["subset"]
            if dataSet != data_type:
                continue
            try:
                with open("/mnt/BROAD-datasets/video/" + dataSet + "/image_resnet50_feature/" + str(video) + ".pkl", 'rb') as f:
                    image_fea = np.array(cPickle.load(f))
                with open("/mnt/BROAD-datasets/video/" + dataSet + "/audio_feature/" + str(video) + ".pkl", 'rb') as f:
                    audio_fea = np.array(cPickle.load(f))

            except:
                continue
            print index, video
            index += 1

            audio_fea = audio_fea[:np.shape(image_fea)[0]]
            image_fea = image_fea[:np.shape(audio_fea)[0]]
            video_fea = np.append(image_fea, audio_fea, axis=1)
            video_len = np.shape(video_fea)[0]
            this_vid_proposals = []
            inputs = []
            for pos in range(0, video_len - window_lens[0], window_stride):
                inputs.append([video_fea[pos:pos + window_lens[0]]])
            probs = paddle.infer(
                output_layer=prob_layer, parameters=parameters, input=inputs, field="value")
            for stride_index, prob in enumerate(probs):
                pos = stride_index * window_stride
                score = int(prob[1] * 100) / 100.0
                if score == 0.0:
                    continue
                proposal = {
                        'score': int(prob[1] * 100) / 100.0,
                        'segment': [pos, pos + window_lens[0]],
                        }
                this_vid_proposals += [proposal]
            if  not proposal_data['results'].has_key(video): 
                proposal_data['results'][video] = this_vid_proposals
            else:
                proposal_data['results'][video] += this_vid_proposals

    window_lens = [80, 100, 120, 150, 180, 240]
    window_strides = [9, 12, 16, 16, 16, 16, 16]
    proposal_data = {'results': {}, 'version': "VERSION 1.0"}
    for index, window_len in enumerate(window_lens):
        _infer(data_type, window_len, proposal_data, window_strides[index])
        with open("/home/kesci/work/res/" + data_type + "_av_.json", 'w') as fobj:
            json.dump(proposal_data, fobj)

def nms_detections(props, scores, overlap=0.7):
    props = np.array(props)
    scores = np.array(scores)
    t1 = props[:, 0]
    t2 = props[:, 1]
    ind = np.argsort(scores)
    area = (t2 - t1 + 1).astype(float)
    pick = []
    while len(ind) > 0:
        i = ind[-1]
        pick.append(i)
        ind = ind[:-1]
        tt1 = np.maximum(t1[i], t1[ind])
        tt2 = np.minimum(t2[i], t2[ind])
        wh = np.maximum(0., tt2 - tt1 + 1.0)
        o = wh / (area[i] + area[ind] - wh)
        ind = ind[np.nonzero(o <= overlap)[0]]
    nms_props, nms_scores = props[pick,: ], scores[pick]
    return nms_props, nms_scores

def refine(data_type):
    proposal_data = {'results': {}, 'version': "VERSION 1.0"}
    json_data = load_json("res/" + data_type + ".json")
    database = json_data['results']
    for video in database.keys():
        this_vid_proposals = []
        probs = []
        scores = []
        for index, i in enumerate(database[video]):
            st, ed = i['segment']
            score = i['score']
            probs.append([st, ed])
            scores.append(i['score'])
        if len(probs) == 0:
            continue
        nms_props, nms_scores = nms_detections(probs, scores, 0.05)
        for prob, score in zip(nms_props, nms_scores):
            proposal = {
                    'score': score,
                    'segment': [prob[0], prob[1]],
                   }
            this_vid_proposals += [proposal]
        proposal_data['results'][video] = this_vid_proposals
    with open("res/" + data_type + "_refine.json", 'w') as fobj:
        json.dump(proposal_data, fobj)

train()
infer('validataion')
refine('validation_av_')
