# encoding=utf8
"""268"""
import codecs

seg_reader = codecs.open('test_segs', 'r', 'utf8')
rst_reader = codecs.open('test_words.csv', 'r', 'utf8')
writer = codecs.open('test_mapped.csv', 'w', 'utf8')

for seg_line in seg_reader:
    rst_line = rst_reader.readline()
    segs = seg_line.strip().split('|')
    topics, emotions, ranks = map(lambda x: x.split(';'), rst_line.strip().split(','))
    map_topics, map_emotions, map_ranks = [], [], []
    for idx, emotion in enumerate(emotions):
        if len(emotion) == 0:
            continue
        rank = ranks[idx]
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
                map_rank = int(rank)
                map_emotion = emotion
                if map_rank != 0:
                    for k in xrange(i - 1, -1, -1):
                        if i - k >= 3:
                            break
                        if not all(map(lambda x: 0x4e00 <= ord(x) <= 0x9fa5, segs[k])):
                            break
                        if segs[k] in topics or segs[k] in emotions:
                            break
                        if segs[k] == u'不':
                            map_emotion = u'不' + map_emotion
                            map_rank = 0 - map_rank
                            break
                map_topics.append(topic)
                map_emotions.append(map_emotion)
                map_ranks.append(str(map_rank))
    if len(map_topics) != len(map_emotions) or len(map_emotions) != len(map_ranks):
        print 'Invalid mapping: ' + str(map_topics) + ' ' + str(map_topics) + ' ' + str(map_ranks)
    if len(map_topics) > 0:
        writer.write(';'.join(map_topics) + ';,')
        writer.write(';'.join(map_emotions) + ';,')
        writer.write(';'.join(map_ranks) + ';\n')
    else:
        writer.write('\n')
