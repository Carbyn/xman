#!/usr/bin/env python
"""Divide features by class."""
import sys

if len(sys.argv) != 3:
    print('divide_class.py SRC_PATH DST_PATH')
    exit(-1)

input_file = sys.argv[1]
output_file = sys.argv[2]

outputs = {}
with open(input_file) as reader:
    for line in reader:
        data = line.strip().split('\t')
        if data[-1] not in outputs:
            outputs[data[-1]] = open(output_file + str(data[-1]), 'w')
        outputs[data[-1]].write('\t'.join(data[:-1]) + '\n')
