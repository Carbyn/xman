# -*- coding: utf-8 -*-

import codecs
import os
import re
from pyltp import Segmentor

LTP_DATA_DIR = 'ltp_data'
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
segmentor = Segmentor()
segmentor.load_with_lexicon(cws_model_path, 'lexicon')

with codecs.open('segs', 'w', 'utf8') as writer:
    with codecs.open('train.csv', 'r', 'utf8') as reader:
        for line in reader:
            line = line.strip().split('\t')
            line[1] = re.sub(u'\s+', ',', line[1])
            segs = segmentor.segment(line[1].encode('utf8'))
            writer.write('|'.join(segs).decode('utf8') + '\n')

with codecs.open('test_segs', 'w', 'utf8') as writer:
    with codecs.open('test.csv', 'r', 'utf8') as reader:
        for line in reader:
            line = line.strip().split('\t')
            line[1] = re.sub(u'\s+', ',', line[1])
            segs = segmentor.segment(line[1].encode('utf8'))
            writer.write('|'.join(segs).decode('utf8') + '\n')
segmentor.release()
