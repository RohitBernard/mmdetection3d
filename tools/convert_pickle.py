import json
import os
import numpy as np
import sys
import pickle
from tools.data_converter import kitti_converter as kitti

f = open(sys.argv[1])
custom_data = json.load(f)


complete_data = []
for i, data in enumerate(custom_data["data"]):
    data_dict = {}
    data_dict["image"] = {}
    data_dict["calib"] = {}
    data_dict["annos"] = {}
    name = ["Pedestrian"]
    truncated = []
    occluded = []
    bboxes = []
    dimensions = []
    locations = []
    rotation_y = []
    alpha = []
    data_dict["image"]["image_idx"] = i
    data_dict["image"]["image_path"] = "Data/siemens_factory" + data["filename"]
    data_dict["image"]["image_shape"] = np.array(data["image_shape"]) 
    calib3x3 = np.array(data["camera_intrinsic"]).reshape(3, 3)
    calib4x4 = np.vstack((np.hstack((calib3x3,np.array([[0,0,0]]).T)),np.array([[0,0,0,1]])))
    data_dict["calib"]["P2"] = calib4x4
    data_dict["calib"]["P0"] = calib4x4
    data_dict["calib"]["Tr_imu_to_velo"] = None
    data_dict['calib']['Tr_velo_to_cam'] = None
    data_dict['calib']['R0_rect'] = None
    nHumans = len(data["humans"])
    data_dict["annos"]["name"] = np.array((name * nHumans))
    humansPresent = len(data['humans'])>0
    for object in data["humans"]:
        truncated.append(object["truncated"])
        occluded.append(object["occluded"])
        bboxes.append(object["bbox"])
        dimension = [object["width"], object["height"], object["depth"]]
        dimensions.append(dimension)
        location = [object["x"], object["y"], object["z"]]
        locations.append(location)
        rotation_y.append(object["yaw"])
        alpha.append(-np.arctan2(object['x'], object['z']) + object['yaw'])
    data_dict["annos"]["truncated"] = np.array(truncated)
    data_dict["annos"]["occluded"] = np.array(occluded)
    data_dict["annos"]["bbox"] = np.array(bboxes) if humansPresent else np.zeros([0,4])
    data_dict["annos"]["dimensions"] = np.array(dimensions) if humansPresent else np.zeros([0,3])
    data_dict["annos"]["location"] = np.array(locations) if humansPresent else np.zeros([0,3])
    data_dict["annos"]["rotation_y"] = np.array(rotation_y)
    data_dict["annos"]["alpha"] = np.array(alpha)

    complete_data.append(data_dict)
# print(complete_data)


path = 'mmdetection3d/data/siemens_factory'
with open(os.path.join(path,sys.argv[2]), 'wb') as f:
    pickle.dump(complete_data, f)
info_train_path = os.path.join(path, sys.argv[2])

root_path = ""
kitti.export_2d_annotation(root_path, info_train_path)


