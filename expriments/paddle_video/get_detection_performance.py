import argparse
import numpy as np
import zipfile
import json
import sys
import os

from eval_detection import ANETdetection


def main(ground_truth_filename, prediction_filename,
         subset='testing', tiou_thresholds=np.linspace(0.5, 0.95, 10),
         verbose=False, check_status=True):

    anet_detection = ANETdetection(ground_truth_filename, prediction_filename,
                                   subset=subset, tiou_thresholds=tiou_thresholds,
                                   verbose=verbose, check_status=False)
    anet_detection.evaluate()
    return anet_detection.mAP.mean()


def parse_input():
    description = ('This script allows you to evaluate the ActivityNet '
                   'detection task which is intended to evaluate the ability '
                   'of  algorithms to temporally localize activities in '
                   'untrimmed video sequences.')
    p = argparse.ArgumentParser(description=description)
    p.add_argument('ground_truth_filename',
                   help='Full path to json file containing the ground truth.')
    p.add_argument('prediction_filename',
                   help='Full path to json file containing the predictions.')
    p.add_argument('--subset', default='validation',
                   help=('String indicating subset to evaluate: '
                         '(training, validation)'))
    p.add_argument('--tiou_thresholds', type=float, default=np.linspace(0.5, 0.95, 10),
                   help='Temporal intersection over union threshold.')
    p.add_argument('--verbose', type=bool, default=False)
    p.add_argument('--check_status', type=bool, default=True)
    return p.parse_args()

if __name__ == '__main__':
    args = parse_input()

    json_data = {}
    json_data['errorCode'] = 0
    json_data['errorMsg'] = 'success.'
    json_data['data'] = []

    try:
        mAP = main(**vars(args))
        res = {'name': 'mAP [0.5-0.95]', 'value': mAP}
        json_data['data'].append(res)
        print json.dumps(json_data)
    except:
        json_data['errorCode'] = 1
        json_data['errorMsg'] = 'evaluate failed.'
        print json.dumps(json_data)
        sys.exit(1)
