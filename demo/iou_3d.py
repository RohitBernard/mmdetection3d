# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import json
import cv2
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
from mmcv.parallel import collate, scatter
from mmdet3d.apis import init_model
from mmdet3d.core import (Box3DMode, CameraInstance3DBoxes)
from mmdet3d.core.bbox import get_box_type
from mmdet3d.datasets.pipelines import Compose
from copy import deepcopy

CAM_INTRINSIC = [[739.0083618164062, 0.0, 640.0],
                 [0.0, 623.5382080078125, 360.0],
                 [0.0, 0.0, 1.0]]

CAM_EXTRINSIC = [[-1.0, 0.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0, 1.0],
                 [0.0, 0.0, 1.0, 25.0],
                 [0.0, 0.0, 0.0, 1.0]]

def main():

    parser = ArgumentParser()
    # parser.add_argument('image', help='image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--axis_aligned',
        action='store_true',
        help='yaw is set to zero')

    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    #####
    # RUN INFERENCE ON A SET OF IMAGES
    #####

    with open('mmdetection3d/data/siemens_factory/siemens_val.pkl', 'rb') as f:
        data = pickle.load(f)

    # corners_3d_boxes = []
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    thresh = 0.5
    thresholds = [0.3, 0.5, 0.7, 0.9]
    pr_data = []
    n_groundtruths = 0
    for d in data:
        boxes_3d_gt = []
        for j in range(len(d['annos']['name'])):
            box_3d_gt = []
            box_3d_gt.append(d['annos']['location'][j][0])
            box_3d_gt.append(d['annos']['location'][j][1])
            box_3d_gt.append(d['annos']['location'][j][2])
            box_3d_gt.append(d['annos']['dimensions'][j][0])
            box_3d_gt.append(d['annos']['dimensions'][j][1])
            box_3d_gt.append(d['annos']['dimensions'][j][2])
            box_3d_gt.append(d['annos']['rotation_y'][j] if not args.axis_aligned else 0)
            boxes_3d_gt.append(box_3d_gt)
        boxes_3d_gt = CameraInstance3DBoxes(boxes_3d_gt)
        img = cv2.imread(d["image"]['image_path'])
        result, inf_data = inference(model, img)
        boxes_3d_preds = result[0]['img_bbox']['boxes_3d']
        if args.axis_aligned:
            axis_aligned_tensor = boxes_3d_preds.tensor
            axis_aligned_tensor[:,-1] = 0
            boxes_3d_preds = CameraInstance3DBoxes(axis_aligned_tensor)

        confidence_scores = result[0]['img_bbox']['scores_3d'].numpy()
        iou = CameraInstance3DBoxes.overlaps(boxes_3d_gt, boxes_3d_preds).numpy()
        for i in range(len(confidence_scores)):
            pr_dict = {}
            iou_score = max(iou[:,i]) if (len(iou[:,i]) != 0) else 0
            pr_dict["score"] = confidence_scores[i]
            pr_dict["iou_score"] = iou_score
            pr_data.append(pr_dict)

        n_groundtruths += iou.shape[0]
        # print("iou", iou)
        rows = len(boxes_3d_gt.tensor)
        cols = iou.shape[1]
        # print(rows, cols)
        # print(iou.shape)
        TP = (np.sum(np.amax(iou, axis=0) > np.full(cols, thresh))) if (cols > 0 and rows > 0) else 0
        if TP > rows:
            TP = rows
        FN = max((rows - cols),0)
        FP = cols - TP
        # print(boxes_3d_gt)
        # print(boxes_3d_preds)
        # print(iou)
        # print(TP,FP,FN)
        true_positives += TP
        false_negatives += FN
        false_positives += FP

    accuracy = true_positives / (true_positives + false_negatives + false_positives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / n_groundtruths

    pr_data = sorted(pr_data, key=lambda x: x['score'], reverse=True) 

    precs = []
    recs = []

    for threshold in thresholds:
        cum_tp = 0
        cum_fp = 0
        precisions = []
        recalls = []
        for pr in pr_data:
            if pr["iou_score"] > threshold:
                cum_tp += 1
            else:
                cum_fp += 1
            prec = cum_tp / (cum_tp + cum_fp)
            rec = cum_tp / n_groundtruths
            precisions.append(prec)
            recalls.append(rec)

        precs.append(prec)
        recs.append(rec)

        ap11_bins = []
        for i in np.arange(0,1.1,0.1):
            idx = None
            for j in range(len(recalls)):
                if recalls[j]>=i:
                    idx = j
                    break
            ap11_bins.append(max(precisions[idx:]) if idx != None else 0)
        ap11 = (sum(ap11_bins)/11)*100
        print('AP@11 at threshold {0}:'.format(threshold), ap11)

        ap40_bins = []
        for i in np.arange(0.025,1.025,0.025):
            idx = None
            for j in range(len(recalls)):
                if recalls[j]>=i:
                    idx = j
                    break
            ap40_bins.append(max(precisions[idx:]) if idx != None else 0)
        ap40 = (sum(ap40_bins)/40)*100
        print('AP@40 at threshold {0}:'.format(threshold), ap40)
        print("---"*30)
        plot_pr_curve(recalls, precisions, threshold)
    plot_pt_curve(precs, thresholds)
    plot_rt_curve(recs, thresholds)


    # print(f"True Positives : {true_positives}\n\
    #     False Positives : {false_positives}\n\
    #     False Negatives : {false_negatives}\n\
    #     Precision : {precision}\n\
    #     Recall : {recall}\n\
    #     Accuracy = {accuracy*100}%")


def plot_pt_curve(precisions, thresholds):
    plt.plot(thresholds, precisions)
    plt.xlabel("Threshold")
    plt.ylabel("Precision")
    plt.title("Precision vs Threshold Curve")
    plt.show()


def plot_rt_curve(recalls, thresholds):
    plt.plot(thresholds, recalls)
    plt.xlabel("Threshold")
    plt.ylabel("Recall")
    plt.title("Recall vs Threshold Curve")
    plt.show()


def plot_pr_curve(recalls, precisions, threshold):
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title("PR Curve at threshold={0}".format(threshold))
    plt.show()


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
