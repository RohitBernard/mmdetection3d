# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import json
import cv2
import pickle
import numpy as np
import torch
from mmcv.parallel import collate, scatter

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
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    with open('saved_model.pkl', 'rb') as f:
        parameters = pickle.load(f)
    model = parameters['model']
    test_pipeline = parameters['test_pipeline']
    box_type_3d = parameters['box_type_3d']
    box_mode_3d = parameters['box_mode_3d']

    out_data = {"data":[]}
    root = ""
    with open('mmdetection3d/data/siemens_factory/siemens_val.pkl', 'rb') as f:
        data = pickle.load(f)
    out = cv2.VideoWriter('output_video_with_gt.avi',cv2.VideoWriter_fourcc(*'DIVX'), 8, (1280,720))

    for d in data:
        boxes_3d_gt = []
        img = cv2.imread(root+d["image"]['image_path'])
        result, inf_data = inference(model, img, test_pipeline, box_mode_3d, box_type_3d)
        inferences = deepcopy(result[0]['img_bbox']['boxes_3d'].tensor).numpy()
        cam_extrinsic = np.array(CAM_EXTRINSIC)
        for i in range(len(inferences)):
            pos = inferences[i,:3]
            yaw = inferences[i,-1]
            pos = np.hstack((pos,[1]))
            pos *= [1,-1,-1,1]
            worldPos = cam_extrinsic.dot(pos)
            worldPos /= worldPos[-1]
        draw_bboxes(img, result[0]['img_bbox']['boxes_3d'], CAM_INTRINSIC, color=(0, 0, 255))
        if cv2.waitKey(0) == ord('q'):
            break

    with open('inferences.json','w') as f:
        json.dump(out_data, f)


def inference(model, image, test_pipeline, box_mode_3d, box_type_3d):
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
