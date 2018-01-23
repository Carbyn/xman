import cPickle
import sys
import json
import pandas as pd
import numpy as np
import gzip
import os


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def getLabelSet(data_type):
    json_data = load_json("data/meta.json")
    database = json_data['database']
    data_dict = {}
    for video_name in database.keys():
        video_info = database[video_name]
        video_subset = video_info["subset"]
        if video_subset == data_type:
            for index, item in enumerate(video_info['annotations']):
                video_info['annotations'][index] = item.values()[0]
            data_dict[video_name] = video_info['annotations']
    return data_dict


def getLabel(video, pos_st, pos_ed, data_dict):
    for st, ed in data_dict[video]:
        if pos_st >= st and pos_ed <= ed:
            return 1
        if pos_st >= st and pos_st <= ed:
            return 2 
        if pos_ed >= st and pos_ed <= ed:
            return 2 
    return 0


def reader_creator(data_type, window_len=0, class_num=2):
    def reader():
        data_dict = getLabelSet(data_type)
        json_data = load_json("data/meta.json")
        database = json_data['database']
        for video in database.keys():
            dataSet = database[video]["subset"]
            if dataSet != data_type:
                continue
            try:
                with open("data/" + dataSet + "/" + str(video) + ".pkl", 'rb') as f:
                    video_fea = cPickle.load(f)
            except:
                continue
            for st, ed in data_dict[video]:
                yield video_fea[int(st):int(ed)], 1
            min_len = 120
            max_len = 300 
            last_end = 0
            for st, ed in data_dict[video]:
                if (st - min_len - 1 <= last_end):
                    continue
                fake_st = np.random.randint(last_end, st - min_len)
                fake_ed = np.random.randint(fake_st + min_len, min(st, fake_st + max_len))
                yield video_fea[int(fake_st):int(fake_ed)], 0
                last_end = ed       

    def window_reader():
        json_data = load_json("data/meta.json")
        database = json_data['database']
        data_dict = getLabelSet(data_type)
        for video in database.keys():
            dataSet = database[video]["subset"]
            if dataSet != data_type:
                continue
            try:
                with open("data/" + dataSet + "/" + str(video) + ".pkl", 'rb') as f:
                    video_fea = cPickle.load(f)
            except:
                continue
            for pos in range(0, (np.shape(video_fea)[0] - window_len), 30):
                label = getLabel(video, pos, pos + window_len, data_dict) 
                if class_num <= 2:
                    label = 0 if label > 1 else label
                yield video_fea[pos:pos + window_len], label 
    if window_len > 0:
        return window_reader
    return reader
