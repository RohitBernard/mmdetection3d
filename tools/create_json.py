import json
import os
import sys

filepath = "/mnt/sdb1/siemens/annos"
train_info = {}
val_info = {}
train_info["data"] = []
val_info["data"] = []

for file in os.listdir(filepath):
    f = open(os.path.join(filepath, file), "rb")
    custom_data = json.load(f)
    if file[-6] in ['6','7']:
        val_info["data"] = val_info["data"] + custom_data["data"]
    else:
        train_info["data"] = train_info["data"] + custom_data["data"]

with open(sys.argv[1], "w") as f:
    json.dump(train_info, f)


with open(sys.argv[2], "w") as f:
    json.dump(val_info, f)





























