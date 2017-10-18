#!/usr/bin/env python
"""Extract features

Note that the input must be sorted.
"""
import sys

if len(sys.argv) != 3:
    print('extract_features.py SRC_PATH DST_PATH')
    exit(-1)

input_file = sys.argv[1]
output_file = sys.argv[2]


def clear():
    """Clear statistic values."""
    global last_bid, last_time
    global page_view, click_num, status_4xx_num, referer_num
    global intervals, head_num, main_num, api_num
    last_bid, last_time = '', 0
    page_view, click_num, status_4xx_num, referer_num = 0, 0, 0, 0
    intervals, head_num, main_num, api_num = [], 0, 0, 0


def extract():
    """Extract features from statistic values.

    Return:
        An array of features.
    """
    global spider
    global page_view, click_num, status_4xx_num, referer_num
    global intervals, head_num, main_num, api_num
    features = []
    # 1. Page view
    value = page_view / 1000.0
    if value > 1.0:
        value = 1.0
    features.append(value)
    # 2. Click num / Page view
    features.append(1.0 * click_num / page_view)
    # 3. 4xx num / Page view
    features.append(1.0 * status_4xx_num / page_view)
    # 4. Referer num / Page view
    features.append(1.0 * referer_num / page_view)
    # 5. Average interval / 3600
    if len(intervals) == 0:
        value = 1.0
    else:
        value = 1.0 * sum(intervals) / len(intervals)
        if value > 1.0:
            value = 1.0
    features.append(value)
    # 6. Head num / Page view
    features.append(1.0 * head_num / page_view)
    # 7. Index num / Page view
    features.append(1.0 * main_num / page_view)
    # 8. API num / Page view
    features.append(1.0 * api_num / page_view)
    # 9. Spider
    features.append(spider)
    # End
    features = map(str, features)
    return features

clear()
with open(output_file, 'w') as writer:
    with open(input_file) as reader:
        line_num = 0
        for line in reader:
            bid, timestamp, method, status, url, referer, spider = line.strip().split('\t')
            if bid != last_bid and page_view != 0:
                writer.write('\t'.join(extract()) + '\n')
                clear()
            last_bid = bid
            page_view += 1
            if url == '/j.php':
                click_num += 1
            if status[0] == '4':
                status_4xx_num += 1
            if referer == '1':
                referer_num += 1
            if last_time != 0:
                intervals.append(int(timestamp) - int(last_time))
            last_time = timestamp
            if method == 'HEAD':
                head_num += 1
            if url == '/':
                main_num += 1
            if url[:11] == '/hao123_api':
                api_num += 1
            line_num += 1
            if line_num % 10000 == 0:
                print(line_num)
        if page_view != 0:
            writer.write('\t'.join(extract()) + '\n')
