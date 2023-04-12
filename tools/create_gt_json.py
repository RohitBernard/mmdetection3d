import os
import json
import sys

f = open(sys.argv[1])
gt_data = json.load(f)


with open(sys.argv[1], "r") as jsonFile:
    gt_data = json.load(jsonFile)
    for d in gt_data['data']:
        for human in d['humans']:
            human['y'] = 1.05


with open(sys.argv[2], "w") as jsonFile:
    json.dump(gt_data, jsonFile)
