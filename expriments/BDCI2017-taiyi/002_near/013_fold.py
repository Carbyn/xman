"""10-fold validation"""
import os
import shutil
import codecs

lines = []
with codecs.open('train_origin.csv', 'r', 'utf8') as reader:
    header = reader.readline()
    for line in reader:
        lines.append(line)

for i in xrange(10):
    print('Round: ' + str(i))
    with codecs.open('train.csv', 'w', 'utf8') as train_writer:
        train_writer.write(header)
        with codecs.open('test.csv', 'w', 'utf8') as test_writer:
            for j in xrange(len(lines)):
                if j % 10 == i:
                    test_writer.write(','.join(lines[j].split('\t')[:2]) + '\n')
                else:
                    train_writer.write(lines[j])
    print('    000_words')
    os.system('python 000_words.py >> logs')
    print('    001_cut')
    os.system('python 001_cut.py >> logs 2>&1')
    print('    003_classify')
    os.system('python 003_classify.py >> logs')
    print('    004_predict')
    os.system('python 004_predict.py >> logs')
    print('    005_near')
    os.system('python 005_near.py >> logs')
    shutil.copy('test_mapped.csv', 'test_mapped_' + str(i) + '.csv')
