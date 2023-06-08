# Copyright (c) OpenMMLab. All rights reserved.
import ipdb
from argparse import ArgumentParser
import json
import cv2
import os
import time
import pickle
import math
import numpy as np
import torch
from mmcv.parallel import collate, scatter

from mmdet3d.apis import init_model, show_result_meshlab
from mmdet3d.core import (Box3DMode, CameraInstance3DBoxes)
from mmdet3d.core.bbox import get_box_type
from mmdet3d.datasets.pipelines import Compose
from copy import deepcopy

CAM_INTRINSIC = [[739.0083618164062, 0.0, 640.0],
                 [0.0, 623.5382080078125, 360.0],
                 [0.0, 0.0, 1.0]]

CAM_EXTRINSIC = [[-1.00000,	0.00000,	0.00000,	0.00000],
                 [0.00000,	0.81915,	0.57358,	10.00000],
                 [0.00000,	-0.57358,	0.81915,	25.00000],
                 [0.00000,	0.00000,	0.00000,	1.00000]]


def main():

    parser = ArgumentParser()
    # parser.add_argument('image', help='image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.15, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show online visualization results')
    parser.add_argument(
        '--with_offset',
        action='store_true',
        help='show gt bboxes with offset')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visualization results')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    #####
    # RUN INFERENCE ON GT Data to send inference to consistency checker
    #####

    root = "Data/GT_Data_latest"
    with open(os.path.join(root, "annos.json"), 'rb') as f:
        data = json.load(f)

    out_data = {"data":[]}

    for d in data['data']:
        boxes_3d_gt = []
        # if 'Camera3' not in d["image"]['image_path']:
        #     continue
        for j in range(len(d['humans'])):
            worldToCam = np.linalg.inv(CAM_EXTRINSIC)
            cam_pos = np.dot(worldToCam, [d['humans'][j]['x'], d['humans'][j]['y'], d['humans'][j]['z'], 1])
            cam_pos /= cam_pos[3]
            box_3d_gt = []
            box_3d_gt.append(cam_pos[0])#d['humans'][j]['x'])
            box_3d_gt.append(1.05)#d['humans'][j]['y'] if not args.with_offset else 1.05)
            box_3d_gt.append(-cam_pos[2])#d['humans'][j]['z'])
            box_3d_gt.append(d['humans'][j]['width'])
            box_3d_gt.append(d['humans'][j]['height'])
            box_3d_gt.append(d['humans'][j]['depth'])
            box_3d_gt.append(d['humans'][j]['yaw'])
            boxes_3d_gt.append(box_3d_gt)

        boxes_3d_gt = CameraInstance3DBoxes(boxes_3d_gt)
 
        img = cv2.imread(root+d["filename"])
        result, inf_data = inference(model, img)

        inferences = deepcopy(result[0]['img_bbox']['boxes_3d'].tensor).numpy()

        cam_extrinsic = np.array(CAM_EXTRINSIC)
        for i in range(len(inferences)):
            pos = inferences[i,:3]
            # print("Latest model Predicted of position in camera frame:", pos)
            # yaw = inferences[i,-1]
            pos = np.hstack((pos,[1]))
            pos *= [1,-1,-1,1]
            worldPos = cam_extrinsic.dot(pos)
            worldPos /= worldPos[-1]
            # print("Latest Model prediction of position in world frame", worldPos)
            # worldYaw = cam_yaw - yaw
            # while(worldYaw>math.pi):
            #     worldYaw-=2*math.pi
            # while(worldYaw<-math.pi):
            #     worldYaw+=2*math.pi
            # inferences[i,:] = np.hstack((worldPos[:3],inferences[i,3:6], worldYaw))
            inferences[i,:6] = np.hstack((worldPos[:3],inferences[i,3:6]))
        # if not args.with_offset:
        # draw_bboxes(img, result[0]['img_bbox']['boxes_3d'], CAM_INTRINSIC, color=(0, 0, 255))
        if cv2.waitKey(0) == ord('q'):
            break
        frame = {
            "filename": d["filename"],
            'image_shape': d['image_shape'],
            'humans': []
        }
        for i in inferences:
            human = {
                'x':float(i[0]),
                'y':float(i[1]+i[4]/2),
                'z':float(i[2]),
                'height':float(i[4]),
                'width':float(i[3]),
                'depth':float(i[5]),
                # 'yaw':float(i[6]),
            }
            frame['humans'].append(human)
        out_data['data'].append(frame)

    with open('inferences.json','w') as f:
        json.dump(out_data, f)


def inference(model, image):
    """Inference image with the monocular 3D detector.

    Args:
        model (nn.Module): The loaded detector.
        image (np array): Image (cv2 image).
        img_info (dict): Annotation info.

    Returns:
        tuple: Predicted results and data from pipeline.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = deepcopy(cfg.data.test.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)


    data = dict(
        # img_prefix=osp.dirname(image),
        img_info=dict(),
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[])

    # camera points to image conversion
    if box_mode_3d == Box3DMode.CAM:
        data['img_info'].update(dict(cam_intrinsic=CAM_INTRINSIC))


    data['img'] = image
    data['img_fields'] = ['img']
    data['img_shape'] = image.shape
    data['cam2img'] = data['img_info']['cam_intrinsic']

    # ipdb.set_trace()
    data = test_pipeline(data)

    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device.index])[0]
    else:
        # this is a workaround to avoid the bug of MMDataParallel
        data['img_metas'] = data['img_metas'][0].data
        data['img'] = data['img'][0].data

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result, data


def draw_bboxes(img, raw_preds, cam_intrinsic, color):
    preds = raw_preds.tensor.numpy()
    boxes = raw_preds.corners.numpy()

    for i in range(len(boxes)):
        pos = preds[i,:3]
        corners = boxes[i]
        points_2d = cv2.projectPoints(corners, np.array([0,0,0], dtype=np.float32), np.array([0,0,0], dtype=np.float32), np.array(cam_intrinsic), None)[0].reshape(8,2).astype(int)

        img = cv2.line(img, points_2d[0], points_2d[1], color=color, thickness=2)
        img = cv2.line(img, points_2d[1], points_2d[5], color=color, thickness=2)
        img = cv2.line(img, points_2d[5], points_2d[4], color=color, thickness=2)
        img = cv2.line(img, points_2d[4], points_2d[0], color=color, thickness=2)
        img = cv2.line(img, points_2d[3], points_2d[2], color=color, thickness=2)
        img = cv2.line(img, points_2d[2], points_2d[6], color=color, thickness=2)
        img = cv2.line(img, points_2d[6], points_2d[7], color=color, thickness=2)
        img = cv2.line(img, points_2d[7], points_2d[3], color=color, thickness=2)
        img = cv2.line(img, points_2d[0], points_2d[3], color=color, thickness=2)
        img = cv2.line(img, points_2d[4], points_2d[7], color=color, thickness=2)
        img = cv2.line(img, points_2d[5], points_2d[6], color=color, thickness=2)
        img = cv2.line(img, points_2d[1], points_2d[2], color=color, thickness=2)
    cv2.imshow("boxes", img)


if __name__ == '__main__':
    main()
