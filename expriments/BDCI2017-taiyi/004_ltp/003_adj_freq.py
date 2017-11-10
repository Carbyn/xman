# -*- coding: utf-8 -*-

import codecs
import os
from pyltp import Postagger

LTP_DATA_DIR = 'ltp_data'
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
postagger = Postagger()
postagger.load(pos_model_path)

emotions = {}
with codecs.open('segs', 'r', 'utf8') as reader:
    for line in reader:
        line = map(lambda seg: seg.encode('utf8'), line.strip().split('|'))
        postags = postagger.postag(line)
        i = 0
        for tag in postags:
            if tag == 'a':
                seg = line[i]
                if seg not in emotions:
                    emotions[seg] = 0
                emotions[seg] += 1
            i += 1
postagger.release()

lists = []
for key, val in emotions.items():
    lists.append((val, key))
lists = sorted(lists, reverse=True)

with codecs.open('adj_freq', 'w', 'utf8') as writer:
    for val, key in lists:
        writer.write('%5d %s\n' % (val, key.decode('utf8')))
