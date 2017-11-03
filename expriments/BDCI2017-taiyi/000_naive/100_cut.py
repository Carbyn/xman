import codecs
import jieba

with codecs.open('test_segs', 'w', 'utf8') as writer:
    with codecs.open('test.csv', 'r', 'utf8') as reader:
        for line in reader:
            line = line.strip().split(',')
            segs = jieba.cut(','.join(line[1:]))
            writer.write('|'.join(segs) + '\n')
