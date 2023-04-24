import json
import os
import sys
import random

#### 
# Args order
# 1 - path to dataset directory : string
# 2 - train split % : float [0,1]
# 3 - shuffle : boolean ['True', 'False']
#####

datapath = sys.argv[1]
train_info = {}
val_info = {}
train_info["data"] = []
val_info["data"] = []

with open(os.path.join(datapath, 'annos.json')) as f:
    annos = json.load(f)
    annos = annos['data']

if sys.argv[3] == 'True':
    random.shuffle(annos)

split_idx = int(len(annos) * float(sys.argv[2]))
train_info["data"] = annos[:split_idx]
val_info["data"] = annos[split_idx:]

with open(os.path.join(datapath, 'train.json'), "w") as f:
    json.dump(train_info, f)


with open(os.path.join(datapath, 'val.json'), "w") as f:
    json.dump(val_info, f)
