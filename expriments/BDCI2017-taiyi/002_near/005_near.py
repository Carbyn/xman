"""268"""
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
                k = 1
                left, right = True, True
                while (left or right) and (0 <= i - k or i + k < len(segs)):
                    if 0 <= i - k:
                        if not all(map(lambda x: 0x4e00 <= ord(x) <= 0x9fa5, segs[i - k])):
                            left = False
                        if left:
                            if segs[i - k] in topics:
                                topic = segs[i - k]
                                break
                    if i + k < len(segs):
                        if not all(map(lambda x: 0x4e00 <= ord(x) <= 0x9fa5, segs[i + k])):
                            right = False
                        if right:
                            if segs[i + k] in topics:
                                topic = segs[i + k]
                                break
                    k += 1
                mapped_topics.append(topic)
                break
    if len(mapped_topics) != len(emotions) or len(emotions) != len(ranks):
        print 'Invalid mapping: ' + str(mapped_topics) + ' ' + str(emotions) + ' ' + str(ranks)
    if len(mapped_topics) > 0:
        writer.write(';'.join(mapped_topics) + ';,')
        writer.write(';'.join(emotions) + ';,')
        writer.write(';'.join(ranks) + ';\n')
    else:
        writer.write('\n')
