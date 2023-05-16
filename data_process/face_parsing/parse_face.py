import logging
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from .model import BiSeNet

CURRENT_DIR = os.path.dirname(__file__)

# def vis_parsing_maps(image: np.array,
#                      parsing_anno,
#                      stride,
#                      save_im=False,
#                      save_path='vis_results/parsing_map_on_im.jpg',
#                      img_size=(512, 512)):
#     vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
#     vis_parsing_anno = cv2.resize(vis_parsing_anno,
#                                   None,
#                                   fx=stride,
#                                   fy=stride,
#                                   interpolation=cv2.INTER_NEAREST)
#     vis_parsing_anno_color = np.zeros(
#         (vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + np.array(
#             [255, 255, 255])  # + 255

#     num_of_class = np.max(vis_parsing_anno)
#     # print(num_of_class)
#     for pi in range(1, 14):
#         index = np.where(vis_parsing_anno == pi)
#         vis_parsing_anno_color[index[0], index[1], :] = np.array([255, 0, 0])

#     for pi in range(14, 16):
#         index = np.where(vis_parsing_anno == pi)
#         vis_parsing_anno_color[index[0], index[1], :] = np.array([0, 255, 0])
#     for pi in range(16, 17):
#         index = np.where(vis_parsing_anno == pi)
#         vis_parsing_anno_color[index[0], index[1], :] = np.array([0, 0, 255])
#     for pi in range(17, num_of_class + 1):
#         index = np.where(vis_parsing_anno == pi)
#         vis_parsing_anno_color[index[0], index[1], :] = np.array([255, 0, 0])

#     vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
#     index = np.where(vis_parsing_anno == num_of_class - 1)
#     vis_im = cv2.resize(vis_parsing_anno_color,
#                         img_size,
#                         interpolation=cv2.INTER_NEAREST)
#     if save_im:
#         cv2.imwrite(save_path, vis_im)


def visualize_parsing_maps(bgr_image: np.array, parsing_res: np.array):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 0, 85], [255, 0, 170], [0, 255, 0], [85, 255, 0],
                   [170, 255, 0], [0, 255, 85], [0, 255, 170], [0, 0, 255], [85, 0, 255], [170, 0, 255], [0, 85, 255],
                   [0, 170, 255], [255, 255, 0], [255, 255, 85], [255, 255, 170], [255, 0, 255], [255, 85, 255],
                   [255, 170, 255], [0, 255, 255], [85, 255, 255], [170, 255, 255]]
    assert bgr_image.shape[:2] == parsing_res.shape
    vis_parsing_anno_color = np.zeros((*parsing_res.shape, 3)) + 255

    num_of_class = np.max(parsing_res)

    for pi in range(1, num_of_class + 1):
        index = np.where(parsing_res == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)

    vis_im = cv2.addWeighted(bgr_image, 0.4, vis_parsing_anno_color, 0.6, 0)
    return vis_im


def parse_faces(img_list: list,
                parsing_dir: str = None,
                ckpt: str = os.path.join(CURRENT_DIR, '79999_iter.pth'),
                debug_freq: int = 0,
                debug_dir: str = None) -> list:
    """Parse face and torso."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if parsing_dir is not None:
        logging.info(f'\t-> Parsed images save to {parsing_dir}')

    res = []
    with torch.no_grad():
        for idx, bgr_image in enumerate(tqdm(img_list, desc='Face Parsing')):
            # image = img.resize((512, 512), Image.BILINEAR)
            # image = image.convert("RGB")
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            rgb_image = cv2.resize(rgb_image, (512, 512), interpolation=cv2.INTER_AREA)

            rgb_image = to_tensor(rgb_image).unsqueeze(0)
            rgb_image = rgb_image.to(device)
            out = net(rgb_image)[0]
            parsing_res = out.squeeze(0).cpu().numpy().argmax(0)
            parsing_res = cv2.resize(parsing_res, dsize=bgr_image.shape[:2], interpolation=cv2.INTER_NEAREST)
            if parsing_dir is not None:
                cv2.imwrite(os.path.join(parsing_dir, f'{idx}.jpg'), parsing_res)
            else:
                res.append(parsing_res)
            if debug_freq > 0 and debug_dir is not None and idx % debug_freq == 0:
                os.makedirs(debug_dir, exist_ok=True)
                vis_im = visualize_parsing_maps(bgr_image=bgr_image, parsing_res=parsing_res)
                cv2.imwrite(os.path.join(debug_dir, f'{idx}_parsed.jpg'), vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                cv2.imwrite(os.path.join(debug_dir, f'{idx}.jpg'), bgr_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    return res


class FaceParser:
    def __init__(
            self,
            n_classes: int = 19,
            ckpt: str = os.path.join(CURRENT_DIR, '79999_iter.pth'),
    ) -> None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.device = device

        n_classes = n_classes
        net = BiSeNet(n_classes=n_classes)
        net.to(device)
        net.load_state_dict(torch.load(ckpt, map_location=device))
        net.eval()
        self.net = net

        logging.info(f'\t-> FaceParser loaded from {ckpt}')

    def parse_faces(self, img_list: list, parsing_dir: str = None, debug_freq: int = 0, debug_dir: str = None) -> list:
        """Parse face and torso."""
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        if parsing_dir is not None:
            logging.info(f'\t-> Parsed images save to {parsing_dir}')

        res = []
        with torch.no_grad():
            for idx, bgr_image in enumerate(tqdm(img_list, desc='Face Parsing')):
                # image = img.resize((512, 512), Image.BILINEAR)
                # image = image.convert("RGB")
                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                rgb_image = cv2.resize(rgb_image, (512, 512), interpolation=cv2.INTER_AREA)

                rgb_image = to_tensor(rgb_image).unsqueeze(0)
                rgb_image = rgb_image.to(self.device)
                out = self.net(rgb_image)[0]
                parsing_res = out.squeeze(0).cpu().numpy().argmax(0)
                parsing_res = cv2.resize(parsing_res, dsize=bgr_image.shape[:2], interpolation=cv2.INTER_NEAREST)
                if parsing_dir is not None:
                    cv2.imwrite(os.path.join(parsing_dir, f'{idx}.jpg'), parsing_res)
                else:
                    res.append(parsing_res)
                if debug_freq > 0 and debug_dir is not None and idx % debug_freq == 0:
                    os.makedirs(debug_dir, exist_ok=True)
                    vis_im = visualize_parsing_maps(bgr_image=bgr_image, parsing_res=parsing_res)
                    cv2.imwrite(os.path.join(debug_dir, f'{idx}_parsed.jpg'), vis_im,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    cv2.imwrite(os.path.join(debug_dir, f'{idx}.jpg'), bgr_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        return res
