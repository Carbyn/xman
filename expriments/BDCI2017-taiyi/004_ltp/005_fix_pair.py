# -*- coding: utf-8 -*-

import codecs
import os

topics = {}
with codecs.open('test_predict_new.csv', 'w', 'utf8') as writer:
    with codecs.open('test_predict.csv', 'r', 'utf8') as predict_reader:
        with codecs.open('test_pair', 'r', 'utf8') as pair_reader:
            predict_reader.readline()
            for rline in predict_reader:
                rline = rline.strip().split(',')
                rtopics = rline[2].split(';')
                remotions = rline[3].split(';')

                aline = pair_reader.readline()
                aline = aline.strip().split('\t')

                if len(aline) == 3:
                    atopics = aline[1].split(';')
                    aemotions = aline[2].split(';')

                    i = 0
                    for rtopic in rtopics:
                        if rtopic == 'NULL':
                            j = 0
                            for aemotion in aemotions:
                                if remotions[i] == aemotion:
                                    if len(atopics[j]) > 1:
                                        rtopics[i] = atopics[j]
                                        if atopics[j] not in topics:
                                            topics[atopics[j]] = 0
                                        topics[atopics[j]] += 1
                                j += 1
                        i += 1

                rline[2] = ';'.join(rtopics)
                writer.write(','.join(rline) + '\n')

lists = []
for key, val in topics.items():
    lists.append((val, key))
lists = sorted(lists, reverse=True)

with codecs.open('topic_freq', 'w', 'utf8') as writer:
    for val, key in lists:
        writer.write('%5d %s\n' % (val, key))
