# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

import os
import pprint
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib.models as models
from lib.config import config, update_config
from lib.utils import utils, transforms
from lib.datasets import get_dataset
from lib.core import function
import numpy as np

from PIL import Image, ImageDraw

def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)
    parser.add_argument('--model-file', help='model parameters', required=True, type=str)
    parser.add_argument('--single-image', dest='single_image',
                        action='store_true',)

    args = parser.parse_args()
    update_config(config, args)
    return args


def draw_facial_landmarks(img_meta, landmarks, output_dir):
    with Image.open(img_meta['raw_img'][0]).convert('RGB') as raw_img:
        img = raw_img.copy()
        draw = ImageDraw.Draw(img)
        for point in landmarks[0]:
            draw.regular_polygon(bounding_circle=((float(point[0]), float(point[1])), 0.5), n_sides=32, fill='red')

        img_crop = np.array(img).astype('float32')
        img_crop = transforms.crop(img_crop, img_meta['center'][0], img_meta['scale'][0], [256, 256])
        img_crop = Image.fromarray(np.uint8(img_crop)).convert('RGB')

        raw_img_crop = np.array(raw_img).astype('float32')
        raw_img_crop = transforms.crop(raw_img_crop, img_meta['center'][0], img_meta['scale'][0], [256, 256])
        raw_img_crop = Image.fromarray(np.uint8(raw_img_crop)).convert('RGB')

        raw_img_crop.save(os.path.join(output_dir, 'face_raw.png'))
        img_crop.save(os.path.join(output_dir, 'face.png'))


def main():

    args = parse_args()

    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()
    model = models.get_face_alignment_net(config)

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # load model
    state_dict = torch.load(args.model_file)
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
    else:
        model.module.load_state_dict(state_dict)

    dataset_type = get_dataset(config)

    if args.single_image:
        test_loader = DataLoader(
            dataset=dataset_type(config,
                                is_train=False, raw_img=args.single_image),
            batch_size=1,
            shuffle=True,
            num_workers=config.WORKERS,
            pin_memory=config.PIN_MEMORY,
        )
        img_meta, prediction = function.inference(config, test_loader, model, single_image=args.single_image)
        draw_facial_landmarks(img_meta, prediction, final_output_dir)
    else:
        test_loader = DataLoader(
            dataset=dataset_type(config,
                                is_train=False),
            batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
            shuffle=False,
            num_workers=config.WORKERS,
            pin_memory=config.PIN_MEMORY,
        )
        nme, predictions = function.inference(config, test_loader, model)
        torch.save(predictions, os.path.join(final_output_dir, 'predictions.pth'))


if __name__ == '__main__':
    main()

