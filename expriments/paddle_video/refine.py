import paddle.v2 as paddle
from network import lstm_net
import json
from reader import load_json
import gzip
import math


def infer(data_type):
    proposal_data = {'results': {}, 'version': "VERSION 1.0"}
    json_data = load_json("res/" + data_type + ".json")
    database = json_data['results']
    for video in database.keys():
        new_seg = []
        new_score = []
        this_vid_proposals = [] 
        last_ed = 0 
        last_st = 0
        width = 0
        for index, i in enumerate(database[video]):
            st, ed = i['segment']
            score = i['score']
            if score < 0.5:
                continue
            if st - last_ed < 30 and width <= 6 and last_ed != 0:
                width += 1
                last_ed = ed
                continue
            if last_ed != 0:

                new_seg.append([last_st, last_ed])
                new_score.append(i['score'])
            last_st = st 
            last_ed = ed 
            width = 0

        for i, sore in zip(new_seg, new_score):
            last_st, last_ed = i
            if last_ed - last_st < 150:
                if last_ed - last_st <= 30:
                    continue
                last_st = (last_st + last_ed) / 2 - 80 
                last_ed = (last_st + last_ed) / 2 + 80 

            proposal = {
                    'score': sore,
                    'segment': [last_st, last_ed],
                   }
            this_vid_proposals += [proposal]
        proposal_data['results'][video] = this_vid_proposals
    with open("res/" + data_type + "_refine.json", 'w') as fobj:
        json.dump(proposal_data, fobj)
infer('validation')
#infer('testing')
