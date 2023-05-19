# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import logging
import os

import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as pil
import torch
from torchvision import transforms
import tqdm

from . import networks
from .layers import disp_to_depth


CURRENT_DIR = os.path.dirname(__file__)

def check_ckpt(ckpt):
    source_url = 'https://github.com/harlanhong/CVPR2022-DaGAN#pre-trained-checkpoint'
    encoder_path = os.path.join(ckpt, "encoder.pth")
    depth_decoder_path = os.path.join(ckpt, "depth.pth")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f'{encoder_path} not Found. Please download it from {source_url}')
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f'{depth_decoder_path} not Found. Please download it from {source_url}')

def extract_face_depth(img_list: list,
                       ckpt: str = os.path.join(CURRENT_DIR, 'depth_face_model_Voxceleb2_10w'),
                       num_layers: int = 50,
                       debug: bool = False,
                       save_dir: str = None):
    """Function to predict for a single image or folder of images
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    check_ckpt(ckpt)
    logging.info(f"\t-> Loading pretrained model from {ckpt}")

    encoder_path = os.path.join(ckpt, "encoder.pth")
    depth_decoder_path = os.path.join(ckpt, "depth.pth")

    # LOADING PRETRAINED MODEL
    encoder = networks.ResnetEncoder(num_layers, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    # print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    logging.debug(f"-> Predicting on {len(img_list)} images")

    res = []
    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, bgr_image in enumerate(tqdm.tqdm(img_list, desc='Predict Depth')):
            orignal_size = bgr_image.shape[:2]
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

            # Load image and preprocess
            rgb_image = cv2.resize(rgb_image, (feed_width, feed_height), interpolation=cv2.INTER_AREA)
            rgb_image = transforms.ToTensor()(rgb_image).unsqueeze(0)

            # PREDICTION
            rgb_image = rgb_image.to(device)
            features = encoder(rgb_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(disp,
                                                           size=orignal_size,
                                                           mode="bilinear",
                                                           align_corners=False)

            # Saving numpy file
            _, depth = disp_to_depth(disp_resized, min_depth=0.1, max_depth=100)
            disp_resized_np = depth.squeeze().cpu().numpy()
            res.append(disp_resized_np)

            if debug and save_dir is not None and idx % 10 == 0:
                # Saving colormapped depth image
                vmax = np.percentile(disp_resized_np, 95)
                normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
                mapper = cm.ScalarMappable(norm=normalizer, cmap='rainbow')

                colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
                plt.axis('off')
                plt.imshow(colormapped_im)
                plt.savefig(os.path.join(save_dir, f'{idx}_disp.jpg'))
                plt.clf()

    logging.debug('-> Done!')
    return res


class DaganFaceDepth:
    def __init__(self,
                 num_layers: int = 50,
                 ckpt: str = os.path.join(CURRENT_DIR, 'depth_face_model_Voxceleb2_10w')) -> None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.device = device

        check_ckpt(ckpt)
        logging.debug(f"-> Loading model from {ckpt}")
        encoder_path = os.path.join(ckpt, "encoder.pth")
        depth_decoder_path = os.path.join(ckpt, "depth.pth")

        # LOADING PRETRAINED MODEL
        logging.debug("   Loading pretrained encoder")
        encoder = networks.ResnetEncoder(num_layers, False)
        loaded_dict_enc = torch.load(encoder_path, map_location=device)

        # extract the height and width of image that this model was trained with
        self.feed_height = loaded_dict_enc['height']
        self.feed_width = loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
        encoder.load_state_dict(filtered_dict_enc)
        encoder.to(device)
        encoder.eval()
        self.encoder = encoder
    
         # logging.debug("   Loading pretrained decoder")
        depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

        loaded_dict = torch.load(depth_decoder_path, map_location=device)
        depth_decoder.load_state_dict(loaded_dict)

        depth_decoder.to(device)
        depth_decoder.eval()
        self.depth_decoder = depth_decoder

    def extract_face_depth(self, img_list: list, debug: bool = False, save_dir: str = None):
        """Function to predict for a single image or folder of images
        """
        logging.debug(f"-> Predict depth of {len(img_list)} images")

        res = []
        # PREDICTING ON EACH IMAGE IN TURN
        with torch.no_grad():
            for idx, bgr_image in enumerate(tqdm.tqdm(img_list, desc='Predict Depth')):
                orignal_size = bgr_image.shape[:2]
                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

                # Load image and preprocess
                rgb_image = cv2.resize(rgb_image, (self.feed_width, self.feed_height), interpolation=cv2.INTER_AREA)
                rgb_image = transforms.ToTensor()(rgb_image).unsqueeze(0)

                # PREDICTION
                rgb_image = rgb_image.to(self.device)
                features = self.encoder(rgb_image)
                outputs = self.depth_decoder(features)

                disp = outputs[("disp", 0)]
                disp_resized = torch.nn.functional.interpolate(disp,
                                                               size=orignal_size,
                                                               mode="bilinear",
                                                               align_corners=False)

                # Saving numpy file
                _, depth = disp_to_depth(disp_resized, min_depth=0.1, max_depth=100)
                disp_resized_np = depth.squeeze().cpu().numpy()
                res.append(disp_resized_np)

                if debug and save_dir is not None and idx % 10 == 0:
                    # Saving colormapped depth image
                    vmax = np.percentile(disp_resized_np, 95)
                    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
                    mapper = cm.ScalarMappable(norm=normalizer, cmap='rainbow')

                    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
                    plt.axis('off')
                    plt.imshow(colormapped_im)
                    plt.savefig(os.path.join(save_dir, f'{idx}_disp.jpg'))
                    plt.clf()

        logging.debug('-> Done!')
        return res
