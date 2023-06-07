# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import json
import cv2
import os
import time
import pickle
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
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
                 [0.00000,	0.00000, 	0.00000,	1.00000]]


def main():


    parser = ArgumentParser()
    # parser.add_argument('image', help='image file')
    # parser.add_argument('ann', help='ann file')
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
        '--snapshot',
        action='store_true',
        help='whether to save online visualization results')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    # f = open(args.ann)
    # json_data = json.load(f)

    #####
    # RUN INFERENCE ON A VIDEO
    #####

    # cap = cv2.VideoCapture('/mnt/sdb1/siemens/mmdetection3d/test_siemens/Movie_001.webm')
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     # if frame is read correctly ret is True
    #     if not ret:
    #         print("Can't receive frame (stream end?). Exiting ...")
    #         break
    #     start_time = time.time()
    #     result, data = inference(model, frame, json_data)
    #     print("Inference time is", time.time() - start_time)
    #     draw_bboxes(frame, result, json_data['cam_intrinsic'])
    #     if cv2.waitKey(1) == ord('q'):
    #         break
    # cap.release()
    # cv2.destroyAllWindows()

    #####
    # RUN INFERENCE ON A SET OF IMAGES
    #####

    # root = "Data/Sim_GT_Data"
    # with open(os.path.join(root, "gt_data.json"), 'rb') as f:
    #     data = json.load(f)

    root = ""
    with open('mmdetection3d/data/siemens_factory/siemens_val.pkl', 'rb') as f:
        data = pickle.load(f)

    print("length of the val set", len(data))
    dist_errors = []
    inference_times = []
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    # threshold = 0.60


    for d in data:
        img = cv2.imread(root+d["image"]['image_path'])
        start_time = time.time()
        result, inf_data = inference(model, img)
        inf_time = time.time() - start_time
        inference_times.append(inf_time)
        inferences = deepcopy(result[0]['img_bbox']['boxes_3d'].tensor).numpy()

        ## Thresholding predictions

        # filtered_inferences = np.zeros((0, 3))
        # for i in range(len(result[0]["img_bbox"]["scores_3d"])):
        #     if result[0]["img_bbox"]["scores_3d"][i].item() > threshold:
        #         filtered_inferences = np.vstack((filtered_inferences, inferences[i,:3]))

        gt_positions_data = d['annos']['location']
        inf_positions_data = inferences[:,:3]

        errors = get_min_euclidean_distance(gt_positions_data, inf_positions_data)

        gt_humans = gt_positions_data.shape[0]
        inf_humans = inf_positions_data.shape[0]

        # print('E',errors)
        # print('GT',gt_positions_data)
        # print('INF',inf_positions_data)
        # print('-'*20)
        # print(result[0]["scores_3d"])

        # draw_bboxes(img, result, json_data["viz_intrinsic"])
        # if cv2.waitKey(0) == ord('q'):
        #     break

        true_pos, fal_neg, fal_pos = detection_metric(gt_humans, inf_humans)
        true_positives += true_pos
        false_positives += fal_pos
        false_negatives += fal_neg
        dist_errors = dist_errors + errors 
    
    avg_error = sum(dist_errors)/len(dist_errors)
    accuracy = true_positives / (true_positives + false_negatives + false_positives)
    avg_inf_times = sum(inference_times)/len(inference_times)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    print(f"Avgerage Error = {avg_error}\n\
        True Positives : {true_positives}\n\
        False Positives : {false_positives}\n\
        False Negatives : {false_negatives}\n\
        Precision : {precision}\n\
        Recall : {recall}\n\
        Accuracy = {accuracy*100}%")
    print("Maximum distance error", max(dist_errors))
    print("Minimum distance error", min(dist_errors))
    print("Average Inference time ", avg_inf_times)
    print("Average Inference FPS: ", 1/avg_inf_times)

        
# def draw_bboxes(img, result, cam_intrinsic):
#     raw_preds = result[0]['img_bbox']['boxes_3d']
#     preds = raw_preds.tensor.numpy()
#     boxes = raw_preds.corners.numpy()
#     # print(result[0]['img_bbox']['labels_3d'])
#     for i in range(len(boxes)):
#         pos = preds[i,:3]
#         corners = boxes[i]
#         points_2d = cv2.projectPoints(corners, np.array([0,0,0], dtype=np.float32), np.array([0,0,0], dtype=np.float32), np.array(cam_intrinsic), None)[0].reshape(8,2).astype(int)
#         # print(pos)
#         # print(points_2d[0])
#         img = cv2.line(img, points_2d[0], points_2d[1], color=(0,0,255), thickness=2)
#         img = cv2.line(img, points_2d[1], points_2d[5], color=(0,0,255), thickness=2)
#         img = cv2.line(img, points_2d[5], points_2d[4], color=(0,0,255), thickness=2)
#         img = cv2.line(img, points_2d[4], points_2d[0], color=(0,0,255), thickness=2)
#         img = cv2.line(img, points_2d[3], points_2d[2], color=(0,0,255), thickness=2)
#         img = cv2.line(img, points_2d[2], points_2d[6], color=(0,0,255), thickness=2)
#         img = cv2.line(img, points_2d[6], points_2d[7], color=(0,0,255), thickness=2)
#         img = cv2.line(img, points_2d[7], points_2d[3], color=(0,0,255), thickness=2)
#         img = cv2.line(img, points_2d[0], points_2d[3], color=(0,0,255), thickness=2)
#         img = cv2.line(img, points_2d[4], points_2d[7], color=(0,0,255), thickness=2)
#         img = cv2.line(img, points_2d[5], points_2d[6], color=(0,0,255), thickness=2)
#         img = cv2.line(img, points_2d[1], points_2d[2], color=(0,0,255), thickness=2)
#     cv2.imshow("boxes", img)




def euclidean_distance(x1, y1, z1, x2, y2, z2):
    return math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2) + ((z1 - z2) ** 2))


def get_min_euclidean_distance(gt_positions, inf_positions):
    res = []
    min_no_bbox_gt = min(len(gt_positions), len(inf_positions))
    for gt_pos in gt_positions:
        for inf_pos in inf_positions:
            euc_dist = euclidean_distance(gt_pos[0], gt_pos[1], gt_pos[2], inf_pos[0], inf_pos[1], inf_pos[2])
            res.append(euc_dist)
    res.sort()
    return res[:min_no_bbox_gt]


def flatten(l):
    return [item for sublist in l for item in sublist]


def get_avg_error(gt_positions_data, inf_positions_data):
    min_dists = []
    a = get_min_euclidean_distance(gt_positions_data, inf_positions_data)
    print(a)
    avg_erorr = sum(min_dists) / len(min_dists)
    return avg_erorr


def detection_metric(gt_humans, inf_humans):
    true_pos = 0
    fal_neg = 0
    fal_pos = 0

    true_pos += min(gt_humans, inf_humans)
    if gt_humans > inf_humans:
        fal_neg += gt_humans - inf_humans
    elif gt_humans < inf_humans:
        fal_pos += inf_humans - gt_humans
    return true_pos, fal_neg, fal_pos



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
            


if __name__ == '__main__':
    main()
