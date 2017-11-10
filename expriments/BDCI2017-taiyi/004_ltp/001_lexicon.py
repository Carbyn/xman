# -*- coding: utf-8 -*-

import codecs
import os

lexicons = []
with codecs.open('train.csv', 'r', 'utf8') as reader:
    for line in reader:
        line = line.strip().split('\t')
        if len(line) == 5:
            segs = filter(lambda seg: seg != 'NULL' and seg != '', (line[2] + line[3]).strip().split(';'))
            lexicons.extend(segs)

lexicons = set(lexicons)
with codecs.open('lexicon', 'w', 'utf8') as writer:
    writer.write('\n'.join(lexicons))
