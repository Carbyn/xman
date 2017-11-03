import codecs

nums = {}
with codecs.open('segs', 'r', 'utf8') as reader:
    for line in reader:
        for word in line.strip().split('|'):
            if word not in nums:
                nums[word] = 0
            nums[word] += 1
lists = []
for key, val in nums.items():
    lists.append((val, key))
lists = sorted(lists, reverse=True)
with codecs.open('freq', 'w', 'utf8') as writer:
    for val, key in lists:
        writer.write('%5d %s\n' % (val, key))
