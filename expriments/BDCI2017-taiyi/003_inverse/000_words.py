"""268"""
import codecs

topics, emotions = {}, {}
with codecs.open('train.csv', 'r', 'utf8') as reader:
    reader.readline()
    for line in reader:
        line = line.strip().split('\t')
        if len(line) != 5:
            continue
        for word in line[2].split(';'):
            if word == 'NULL' or len(word) == 0:
                continue
            topics[word] = True
        if len(line[3].split(';')) != len(line[4].split(';')):
            print 'Invalid: ' + line[3] + ' | ' + line[4]
        for word, rank in zip(line[3].split(';'), line[4].split(';')):
            if len(word) == 0:
                continue
            if word not in emotions:
                emotions[word] = 0
            emotions[word] += int(rank)

with codecs.open('topics', 'w', 'utf8') as writer:
    for word in topics.keys():
        if len(word) == 0:
            continue
        writer.write(word + '\n')

with codecs.open('emotions', 'w', 'utf8') as writer:
    for word, rank in emotions.items():
        if len(word) == 0:
            continue
        if rank > 0:
            rank = 1
        elif rank < 0:
            rank = -1
        writer.write(word + ' ' + str(rank) + '\n')
