from collections import defaultdict as dd
import os
import json
import sqlite3
import ast
import re
from sqlite3 import Error
from collections import Counter
import argparse
import glob

import sys
sys.path.append("rat-sql")

def read_json(file):
    with open(file) as json_file:
        json_data = json.load(json_file)
    
    return json_data

def read_jsonl(file):
    with open(file) as json_file:
        json_data = [json.loads(line) for line in json_file]
    
    return json_data

def format_eval(args):
    exp = args.experiment_name
    ratsql_infereval_loc = f'logdir/bert_run/{exp}/ie_dirs/'
    print(ratsql_infereval_loc)
    # step = [1] + [(5000 * x) + 100 for x in range(1, 17)] + [81000]  
    print(os.curdir)   
    eval_folder = ratsql_infereval_loc + '*.eval'
    print(eval_folder)
    eval_list = sorted(glob.glob(eval_folder), key=lambda x: int(x.split('-')[-1][4:-5]))
    
    step_result = {}

    for ratsql_eval in eval_list:
        step = (ratsql_eval.split('step')[-1]).split('.')[0]
        evaluation = read_jsonl(ratsql_eval)

        hardness_cnt = {'easy': 248, 'medium': 446, 'hard': 174, 'extra': 166}

        exact_match = {'easy': 0, 'medium': 0 , 'hard': 0, 'extra': 0}
        overall = 0
        for item in evaluation[0]['per_item']:
            if item['exact']:
                exact_match[item['hardness']] += 1
                overall += 1

        print(f"Step: {step} Result")

        step_result[step] = {}
        for hardness in hardness_cnt.keys():
            step_result[step][hardness] = round(exact_match[hardness] / hardness_cnt[hardness], 4) * 100
            print(f"{hardness}: {exact_match[hardness] / hardness_cnt[hardness]}")
        step_result[step]['overall'] = round(overall / 1034, 4) * 100

        print(f"Overall: {overall / 1034}")

    fileName = f'exp_results/{exp}_result.json'

    with open(fileName, 'w') as fp:
        json.dump(step_result, fp, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Evaluation Folders Location etc')
    parser.add_argument('--experiment_name')

    args = parser.parse_args()

    format_eval(args)
