import codecs

seg_reader = codecs.open('test_segs', 'r', 'utf8')
rst_reader = codecs.open('test_words.csv', 'r', 'utf8')
writer = codecs.open('test_mapped.csv', 'w', 'utf8')

for seg_line in seg_reader:
    rst_line = rst_reader.readline()
    segs = seg_line.strip().split('|')
    topics, emotions, ranks = map(lambda x: x.split(';'), rst_line.strip().split(','))
    mapped_topics = []
    for emotion in emotions:
        for i in xrange(len(segs)):
            if segs[i] == emotion:
                topic = 'NULL'
                for j in xrange(i - 2, i + 3):
                    if 0 <= j < len(segs):
                        if segs[j] in topics:
                            topic = segs[j]
                            break
                mapped_topics.append(topic)
    if len(mapped_topics) > 0:
        writer.write(';'.join(mapped_topics) + ';,')
        writer.write(';'.join(emotions) + ';,')
        writer.write(';'.join(ranks) + ';\n')
    else:
        writer.write('\n')
