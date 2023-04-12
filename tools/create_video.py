import os
import cv2
import json
import sys
import numpy as np
import glob

root = "Data/Sim_GT_Data"
with open(os.path.join(root, "gt_data.json"), 'rb') as f:
    data = json.load(f)

frameSize = (1280, 720)

out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 8, frameSize)

path = 'Data/Sim_GT_Data/CameraCaptures'
for d in data['data']:
    img = cv2.imread(root+d["filename"])
    out.write(img)
print("done")

out.release()
