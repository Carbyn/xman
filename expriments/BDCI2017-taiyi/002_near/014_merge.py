"""merge 10-fold result"""
import codecs

lines = {}
for i in xrange(10):
    with codecs.open('test_mapped_' + str(i) + '.csv', 'r', 'utf8') as reader:
        line_num = 0
        for line in reader:
            lines[line_num * 10 + i] = line
            line_num += 1

with codecs.open('test_mapped.csv', 'w', 'utf8') as writer:
    for key in sorted(lines.keys()):
        writer.write(lines[key])
