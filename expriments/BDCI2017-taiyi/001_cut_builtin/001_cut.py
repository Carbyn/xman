"""268"""
import codecs
import jieba

with codecs.open('topics', 'r', 'utf8') as reader:
    for line in reader:
        jieba.add_word(line.strip())

with codecs.open('emotions', 'r', 'utf8') as reader:
    for line in reader:
        jieba.add_word(line.strip().split(' ')[0])

with codecs.open('segs', 'w', 'utf8') as writer:
    with codecs.open('train.csv', 'r', 'utf8') as reader:
        reader.readline()
        for line in reader:
            line = line.strip().split('\t')
            segs = jieba.cut(line[1])
            writer.write('|'.join(segs) + '\n')

with codecs.open('test_segs', 'w', 'utf8') as writer:
    with codecs.open('test.csv', 'r', 'utf8') as reader:
        for line in reader:
            line = line.strip().split(',')
            segs = jieba.cut(','.join(line[1:]))
            writer.write('|'.join(segs) + '\n')
