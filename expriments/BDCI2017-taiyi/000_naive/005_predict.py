import codecs
from sklearn.externals import joblib

clf = joblib.load('clf.pkl')

vectors = {}
with codecs.open('features', 'r', 'utf8') as reader:
    for line in reader:
        values = line.strip().split(' ')
        word = values[0]
        vector = map(float, values[1:])
        vectors[word] = vector

with codecs.open('test_words.csv', 'w', 'utf-8') as writer:
    line_num = 0
    with codecs.open('test_segs', 'r', 'utf8') as reader:
        for line in reader:
            words = line.strip().split('|')
            labels, X = [], []
            for word in words:
                if word in labels:
                    continue
                if word in vectors:
                    labels.append(word)
                    X.append(vectors[word])
            topics, emotions = [], []
            if len(labels) > 0:
                y = clf.predict(X)
                for i in xrange(len(labels)):
                    if y[i] == 1:
                        topics.append(labels[i])
                    elif y[i] > 1:
                        emotions.append((labels[i], y[i] - 3))
            writer.write(';'.join(topics) + ',')
            writer.write(';'.join(map(lambda x: x[0], emotions)) + ',')
            writer.write(';'.join(map(lambda x: str(x[1]), emotions)) + '\n')
            line_num += 1
            if line_num % 100 == 0:
                print line_num
