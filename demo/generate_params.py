# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import pickle

from mmdet3d.apis import init_model
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
    parser.add_argument('config', help='Config file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    ### Save model and test pipeline
    cfg = model.cfg
    test_pipeline = deepcopy(cfg.data.test.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)

    pickle_obj = {"model": model,
                  "test_pipeline": test_pipeline,
                  "box_type_3d": box_type_3d,
                  "box_mode_3d": box_mode_3d}
    filename = open("saved_model.pkl", 'wb') 
    pickle.dump(pickle_obj, filename)

if __name__ == '__main__':
    main()
