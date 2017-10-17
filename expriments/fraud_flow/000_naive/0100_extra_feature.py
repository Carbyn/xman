#!/usr/bin/env python
"""Parse Nginx logs."""
import sys
import json

LOG_FORMAT = '$remote_addr - $remote_user [$time_local] "$request" ' + \
             '$status $body_bytes_sent "$http_referer" "$http_cookie $sent_http_set_cookie" ' + \
             '"$http_user_agent" $http_host $request_time'

if len(sys.argv) != 3:
    print('parse_log.py SRC_PATH DST_PATH')
    exit(-1)

input_file = sys.argv[1]
output_file = sys.argv[2]

i, formats = 0, []
while i < len(LOG_FORMAT):
    if LOG_FORMAT[i] == '$':
        i += 1
        name = ''
        while i < len(LOG_FORMAT):
            ch = LOG_FORMAT[i]
            if 'a' <= ch <= 'z' or 'A' <= ch <= 'Z' or ch == '_':
                name += ch
                i += 1
            else:
                break
        formats.append(('id', name))
    else:
        formats.append(('ch', LOG_FORMAT[i]))
        i += 1
formats.append(('ch', '\n'))

with open(output_file, 'w') as writer:
    with open(input_file) as reader:
        line_num = 0
        for line in reader:
            valid = True
            i, pos, data = 0, 0, {}
            while i < len(line) and pos < len(formats):
                if formats[pos][0] == 'ch':
                    if line[i] != formats[pos][1]:
                        valid = False
                        break
                    i += 1
                    pos += 1
                else:
                    if pos + 1 >= len(formats) or formats[pos + 1][0] != 'ch':
                        valid = False
                        break
                    value, escaped = '', False
                    while i < len(line):
                        if line[i] == formats[pos + 1][1]:
                            break
                        value += line[i]
                        i += 1
                    data[formats[pos][1]] = value
                    pos += 1
            if i != len(line) or pos != len(formats):
                valid = False
            if valid:
                writer.write(json.dumps(data) + '\n')
            line_num += 1
            if line_num % 10000 == 0:
                print(line_num)
