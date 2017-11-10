# -*- coding: utf-8 -*-

import codecs
import os
from pyltp import Postagger

def find_topic(line, postags, pos):
    l = len(postags)
    # priority: front n > back n > front v > back v
    i = pos
    while (i > 0):
        i = i - 1 
        if postags[i] == 'n':
            return line[i]
        if postags[i] == 'wp':
            break
    i = pos
    while (i < l - 1):
        i = i + 1
        if postags[i] == 'n':
            return line[i]
        if postags[i] == 'wp':
            break
    return 'NULL'
    i = pos
    while (i > 0):
        i = i - 1 
        if postags[i] == 'v':
            return line[i]
        if postags[i] == 'wp':
            break
    i = pos
    while (i < l - 1):
        i = i + 1
        if postags[i] == 'v':
            return line[i]
        if postags[i] == 'wp':
            break
    return 'NULL'

blacklist = set(['完全', '真', '直接', '确实', '一样', '基本', '总之']);

LTP_DATA_DIR = 'ltp_data'
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
postagger = Postagger()
postagger.load(pos_model_path)

with codecs.open('pair', 'w', 'utf8') as writer:
    with codecs.open('segs', 'r', 'utf8') as reader:
        for line in reader:
            line = map(lambda seg: seg.encode('utf8'), line.strip().split('|'))
            postags = postagger.postag(line)
            i = 0
            topics = []
            emotions = []
            for tag in postags:
                if tag == 'a' and line[i] not in blacklist:
                    emotions.append(line[i])
                    topics.append(find_topic(line, postags, i))
                i += 1
            writer.write((''.join(line) + '\t' + ';'.join(topics) + '\t' + ';'.join(emotions)).decode('utf8') + '\n')

with codecs.open('test_pair', 'w', 'utf8') as writer:
    with codecs.open('test_segs', 'r', 'utf8') as reader:
        for line in reader:
            line = map(lambda seg: seg.encode('utf8'), line.strip().split('|'))
            postags = postagger.postag(line)
            i = 0
            topics = []
            emotions = []
            for tag in postags:
                if tag == 'a' and line[i] not in blacklist:
                    emotions.append(line[i])
                    topics.append(find_topic(line, postags, i))
                i += 1
            writer.write((''.join(line) + '\t' + ';'.join(topics) + '\t' + ';'.join(emotions)).decode('utf8') + '\n')
postagger.release()

