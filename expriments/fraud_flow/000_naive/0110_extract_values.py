#!/usr/bin/env python
"""Extract basic values"""
import sys
import json
import re
import time
from datetime import datetime

if len(sys.argv) != 3:
    print('extract_values.py SRC_PATH DST_PATH')
    exit(-1)

input_file = sys.argv[1]
output_file = sys.argv[2]

date_format = '%d/%b/%Y:%H:%M:%S +0800'

with open(output_file, 'w') as writer:
    with open(input_file) as reader:
        line_num = 0
        for line in reader:
            data = json.loads(line.strip())
            extracted = []
            cookie = data['http_cookie'] + '; ' + data['sent_http_set_cookie']
            # 1. BAIDU ID
            result = re.findall(r'BAIDUID=([0-9A-Z]{32})', cookie)
            if len(result) == 0:
                continue
            extracted.append(result[0])
            # 2. Timestamp
            timestamp = time.mktime(datetime.strptime(data['time_local'], date_format).timetuple())
            extracted.append(str(int(timestamp)))
            # 3. Method
            extracted.append(data['request'].split(' ')[0])
            # 4. Status
            extracted.append(data['status'])
            # 5. URL
            extracted.append(data['request'].split(' ')[1].split('?')[0])
            # 6. Referer
            if data['http_referer'] == '-':
                extracted.append('0')
            else:
                extracted.append('1')
            # 7. Spider
            result = re.findall(r'spider|bot', data['http_user_agent'], flags=re.IGNORECASE)
            if len(result) > 0:
                extracted.append('1')
            else:
                extracted.append('0')
            # End
            writer.write('\t'.join(extracted) + '\n')
            line_num += 1
            if line_num % 10000 == 0:
                print(line_num)
