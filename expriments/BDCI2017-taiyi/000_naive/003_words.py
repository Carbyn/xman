import codecs

topics, emotions = {}, {}
with codecs.open('train.csv', 'r', 'utf8') as reader:
    reader.readline()
    for line in reader:
        line = line.strip().split('\t')
        if len(line) != 5:
            continue
        for word in line[2].split(';'):
            if word == 'NULL':
                continue
            topics[word] = True
        for word, rank in zip(line[3].split(';'), line[4].split(';')):
            emotions[word] = rank

with codecs.open('topics', 'w', 'utf8') as writer:
    for word in topics.keys():
        if len(word) == 0:
            continue
        writer.write(word + '\n')

with codecs.open('emotions', 'w', 'utf8') as writer:
    for word, rank in emotions.items():
        if len(word) == 0:
            continue
        writer.write(word + ' ' + rank + '\n')
