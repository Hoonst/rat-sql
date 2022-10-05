from collections import defaultdict as dd
import os
import json
import sqlite3
import ast
import re
from sqlite3 import Error
from collections import Counter

def read_json(file):
    with open(file) as json_file:
        json_data = json.load(json_file)
    
    return json_data

def read_jsonl(file):
    with open(file) as json_file:
        json_data = [json.loads(line) for line in json_file]
    
    return json_data