import codecs
import jieba

with codecs.open('segs', 'w', 'utf8') as writer:
    with codecs.open('train.csv', 'r', 'utf8') as reader:
        reader.readline()
        for line in reader:
            line = line.strip().split('\t')
            segs = jieba.cut(line[1])
            writer.write('|'.join(segs) + '\n')
