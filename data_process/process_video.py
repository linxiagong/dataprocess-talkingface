import argparse
import logging
import os
import tqdm
import cv2
import numpy as np
import glob
import json
import file_ops


def set_logging(save_to_file: bool = False):
    NOCOLOR = "\033[0m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[1;34m"
    # Basic logging set
    handlers = [logging.StreamHandler()]
    if save_to_file:
        logfile = f'./log/data_process/main_process.log'
        os.makedirs(f'./log/data_process/', exist_ok=True)
        handlers += [logging.FileHandler(logfile, 'w')]
    logging.basicConfig(level=logging.INFO,
                        format=f'%(asctime)s - %(filename)s:%(lineno)s: ' + f'{BLUE}%(message)s{NOCOLOR}',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=handlers)


def extract_imgs_from_video(video_file: str, ori_imgs_dir: str = None, resize_shape: tuple = None) -> list:
    """Extract original images for a video file, return a list.
    """
    frame_list = list()
    cap = cv2.VideoCapture(str(video_file))

    for frame_index, (_, frame) in enumerate(iter(cap.read, (False, None))):
        if isinstance(resize_shape, tuple):
            frame = cv2.resize(frame, resize_shape)
            # frame = skimage.transform.resize(frame, resize_shape, preserve_range=True)
        if ori_imgs_dir is not None:
            cv2.imwrite(os.path.join(ori_imgs_dir, f'{frame_index}.jpg'), img=frame)
        frame_list.append(frame.astype(np.uint8))

    cap.release()
    return frame_list


def extract_background(base_dir: str):
    from sklearn.neighbors import NearestNeighbors

    image_paths = glob.glob(os.path.join(base_dir, 'ori_imgs', '*.jpg'))
    # only use 1/20 images to calculate
    image_paths = image_paths[::20]
    # get H, W
    tmp_image = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED)  # [H, W, 3]
    h, w = tmp_image.shape[:2]

    # nearest neighbors
    all_xys = np.mgrid[0:h, 0:w].reshape(2, -1).transpose()
    distss = []
    for image_path in tqdm.tqdm(image_paths, desc='Calculate Background'):
        parse_img = cv2.imread(image_path.replace('ori_imgs', 'parsing'))
        bg = (parse_img[..., 0] == 255) & (parse_img[..., 1] == 255) & (parse_img[..., 2] == 255)
        fg_xys = np.stack(np.nonzero(~bg)).transpose(1, 0)
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fg_xys)
        dists, _ = nbrs.kneighbors(all_xys)
        distss.append(dists)

    distss = np.stack(distss)
    max_dist = np.max(distss, 0)
    max_id = np.argmax(distss, 0)

    bc_pixs = max_dist > 5
    bc_pixs_id = np.nonzero(bc_pixs)
    bc_ids = max_id[bc_pixs]

    imgs = []
    num_pixs = distss.shape[1]
    for image_path in image_paths:
        img = cv2.imread(image_path)
        imgs.append(img)
    imgs = np.stack(imgs).reshape(-1, num_pixs, 3)

    bc_img = np.zeros((h * w, 3), dtype=np.uint8)
    bc_img[bc_pixs_id, :] = imgs[bc_ids, bc_pixs_id, :]
    bc_img = bc_img.reshape(h, w, 3)

    max_dist = max_dist.reshape(h, w)
    bc_pixs = max_dist > 5
    bg_xys = np.stack(np.nonzero(~bc_pixs)).transpose()
    fg_xys = np.stack(np.nonzero(bc_pixs)).transpose()
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fg_xys)
    distances, indices = nbrs.kneighbors(bg_xys)
    bg_fg_xys = fg_xys[indices[:, 0]]
    bc_img[bg_xys[:, 0], bg_xys[:, 1], :] = bc_img[bg_fg_xys[:, 0], bg_fg_xys[:, 1], :]

    cv2.imwrite(os.path.join(base_dir, 'bc.jpg'), bc_img)

    return bc_img


def extract_torso_and_gt(base_dir: str):
    from scipy.ndimage import binary_erosion, binary_dilation
    # load bg
    bg_image = cv2.imread(os.path.join(base_dir, 'bc.jpg'), cv2.IMREAD_UNCHANGED)

    image_paths = glob.glob(os.path.join(base_dir, 'ori_imgs', '*.jpg'))
    for image_path in tqdm.tqdm(image_paths, desc='Extract Torso & GT'):
        # read ori image
        ori_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # [H, W, 3]

        # read semantics
        seg = cv2.imread(image_path.replace('ori_imgs', 'parsing').replace('.jpg', '.png'))
        head_part = (seg[..., 0] == 255) & (seg[..., 1] == 0) & (seg[..., 2] == 0)
        neck_part = (seg[..., 0] == 0) & (seg[..., 1] == 255) & (seg[..., 2] == 0)
        torso_part = (seg[..., 0] == 0) & (seg[..., 1] == 0) & (seg[..., 2] == 255)
        bg_part = (seg[..., 0] == 255) & (seg[..., 1] == 255) & (seg[..., 2] == 255)

        # get gt image
        gt_image = ori_image.copy()
        gt_image[bg_part] = bg_image[bg_part]
        cv2.imwrite(image_path.replace('ori_imgs', 'gt_imgs'), gt_image)

        # get torso image
        torso_image = gt_image.copy()  # rgb
        torso_image[head_part] = bg_image[head_part]
        torso_alpha = 255 * np.ones((gt_image.shape[0], gt_image.shape[1], 1), dtype=np.uint8)  # alpha

        # torso part "vertical" in-painting...
        L = 8 + 1
        torso_coords = np.stack(np.nonzero(torso_part), axis=-1)  # [M, 2]
        # lexsort: sort 2D coords first by y then by x,
        # ref: https://stackoverflow.com/questions/2706605/sorting-a-2d-numpy-array-by-multiple-axes
        inds = np.lexsort((torso_coords[:, 0], torso_coords[:, 1]))
        torso_coords = torso_coords[inds]
        # choose the top pixel for each column
        u, uid, ucnt = np.unique(torso_coords[:, 1], return_index=True, return_counts=True)
        top_torso_coords = torso_coords[uid]  # [m, 2]
        # only keep top-is-head pixels
        top_torso_coords_up = top_torso_coords.copy() - np.array([1, 0])
        mask = head_part[tuple(top_torso_coords_up.T)]
        if mask.any():
            top_torso_coords = top_torso_coords[mask]
            # get the color
            top_torso_colors = gt_image[tuple(top_torso_coords.T)]  # [m, 3]
            # construct inpaint coords (vertically up, or minus in x)
            inpaint_torso_coords = top_torso_coords[None].repeat(L, 0)  # [L, m, 2]
            inpaint_offsets = np.stack([-np.arange(L), np.zeros(L, dtype=np.int32)], axis=-1)[:, None]  # [L, 1, 2]
            inpaint_torso_coords += inpaint_offsets
            inpaint_torso_coords = inpaint_torso_coords.reshape(-1, 2)  # [Lm, 2]
            inpaint_torso_colors = top_torso_colors[None].repeat(L, 0)  # [L, m, 3]
            darken_scaler = 0.98**np.arange(L).reshape(L, 1, 1)  # [L, 1, 1]
            inpaint_torso_colors = (inpaint_torso_colors * darken_scaler).reshape(-1, 3)  # [Lm, 3]
            # set color
            torso_image[tuple(inpaint_torso_coords.T)] = inpaint_torso_colors

            inpaint_torso_mask = np.zeros_like(torso_image[..., 0]).astype(bool)
            inpaint_torso_mask[tuple(inpaint_torso_coords.T)] = True
        else:
            inpaint_torso_mask = None

        # neck part "vertical" in-painting...
        push_down = 4
        L = 48 + push_down + 1

        neck_part = binary_dilation(neck_part,
                                    structure=np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool),
                                    iterations=3)

        neck_coords = np.stack(np.nonzero(neck_part), axis=-1)  # [M, 2]
        # lexsort: sort 2D coords first by y then by x,
        # ref: https://stackoverflow.com/questions/2706605/sorting-a-2d-numpy-array-by-multiple-axes
        inds = np.lexsort((neck_coords[:, 0], neck_coords[:, 1]))
        neck_coords = neck_coords[inds]
        # choose the top pixel for each column
        u, uid, ucnt = np.unique(neck_coords[:, 1], return_index=True, return_counts=True)
        top_neck_coords = neck_coords[uid]  # [m, 2]
        # only keep top-is-head pixels
        top_neck_coords_up = top_neck_coords.copy() - np.array([1, 0])
        mask = head_part[tuple(top_neck_coords_up.T)]

        top_neck_coords = top_neck_coords[mask]
        # push these top down for 4 pixels to make the neck inpainting more natural...
        offset_down = np.minimum(ucnt[mask] - 1, push_down)
        top_neck_coords += np.stack([offset_down, np.zeros_like(offset_down)], axis=-1)
        # get the color
        top_neck_colors = gt_image[tuple(top_neck_coords.T)]  # [m, 3]
        # construct inpaint coords (vertically up, or minus in x)
        inpaint_neck_coords = top_neck_coords[None].repeat(L, 0)  # [L, m, 2]
        inpaint_offsets = np.stack([-np.arange(L), np.zeros(L, dtype=np.int32)], axis=-1)[:, None]  # [L, 1, 2]
        inpaint_neck_coords += inpaint_offsets
        inpaint_neck_coords = inpaint_neck_coords.reshape(-1, 2)  # [Lm, 2]
        inpaint_neck_colors = top_neck_colors[None].repeat(L, 0)  # [L, m, 3]
        darken_scaler = 0.98**np.arange(L).reshape(L, 1, 1)  # [L, 1, 1]
        inpaint_neck_colors = (inpaint_neck_colors * darken_scaler).reshape(-1, 3)  # [Lm, 3]
        # set color
        torso_image[tuple(inpaint_neck_coords.T)] = inpaint_neck_colors

        # apply blurring to the inpaint area to avoid vertical-line artifects...
        inpaint_mask = np.zeros_like(torso_image[..., 0]).astype(bool)
        inpaint_mask[tuple(inpaint_neck_coords.T)] = True

        blur_img = torso_image.copy()
        blur_img = cv2.GaussianBlur(blur_img, (5, 5), cv2.BORDER_DEFAULT)

        torso_image[inpaint_mask] = blur_img[inpaint_mask]

        # set mask
        mask = (neck_part | torso_part | inpaint_mask)
        if inpaint_torso_mask is not None:
            mask = mask | inpaint_torso_mask
        torso_image[~mask] = 0
        torso_alpha[~mask] = 0

        cv2.imwrite(
            image_path.replace('ori_imgs', 'torso_imgs').replace('.jpg', '.png'),
            np.concatenate([torso_image, torso_alpha], axis=-1))


def save_transform(base_dir: str):
    with open(os.path.join(base_dir, 'ori_imgs', 'camera_poses.json'), 'r') as f:
        camera_poses = json.load(f)
    valid_ids = camera_poses.keys()

    # read one image to get H/W
    tmp_image = cv2.imread(os.path.join(base_dir, 'ori_imgs', f'{valid_ids[0]}.jpg'), cv2.IMREAD_UNCHANGED)  # [H, W, 3]
    h, w = tmp_image.shape[:2]

    train_val_split = int(len(valid_ids) * 10 / 11)
    train_ids = valid_ids[:train_val_split]
    val_ids = valid_ids[train_val_split:]

    for split_name, ids in zip(['train', 'val'], [train_ids, val_ids]):
        frames = []
        for i in ids:
            frame_dict = dict(
                img_id=int(i),
                aud_id=int(i),
                transform_matrix=camera_poses[i],
            )
            frames.append(frame_dict)

        transform_dict = dict(
            focal_len=w,
            # focal_len=float(focal_len[0]),
            cx=float(w / 2.0),  # center
            cy=float(h / 2.0),
            frames=frames,
        )
        file_ops.json_dump(transform_dict, os.path.join(base_dir, f'transforms_{split_name}.json'))


def process_video(
    video_file: str,
    base_dir: str,
    task: int = -1,
    resize_shape: tuple = None,
    audio_feat: str = 'hubert',
    debug_freq: int = 0,
):

    # Open the MP4 file
    video = cv2.VideoCapture(str(video_file))
    # Get the total number of frames in the video
    video_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_rate = int(video.get(cv2.CAP_PROP_FPS))
    logging.info(f'Process video {video_file}: Length={video_len}, FPS={fps_rate}. Output to {base_dir}')

    img_list = None
    debug_dir = os.path.join(base_dir, 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    ori_imgs_dir = os.path.join(base_dir, 'ori_imgs')
    os.makedirs(ori_imgs_dir, exist_ok=True)
    parsing_dir = os.path.join(base_dir, 'parsing')
    os.makedirs(parsing_dir, exist_ok=True)
    torso_dir = os.path.join(base_dir, 'torso_imgs')
    os.makedirs(torso_dir, exist_ok=True)
    gt_dir = os.path.join(base_dir, 'gt_imgs')
    os.makedirs(gt_dir, exist_ok=True)
    depth_dir = os.path.join(base_dir, 'depth')
    os.makedirs(depth_dir, exist_ok=True)

    if task == -1 or task == 1:
        logging.info('====== Extract Wav from Video ======')
        from audio_process import extract_wav_from_video
        wav_file = os.path.join(base_dir, 'audio.wav')
        extract_wav_from_video(video_file=video_file, wav_file=wav_file)

    if task == -1 or task == 2:
        logging.info(f'====== Extract Audio Features: {audio_feat} ======')
        if audio_feat == 'hubert':
            from audio_process.hubert import HubertFeatureExtractor
            hubert_extractor = HubertFeatureExtractor()
            hubert_features = hubert_extractor.extract_hubert_features(wav_file,
                                                                       target_len=video_len,
                                                                       target_fps=fps_rate)

            audfeat_file = os.path.join(base_dir, f'audio_{audio_feat}.npy')
            np.save(audfeat_file, hubert_features)
        else:
            raise NotImplementedError(f'Wav processing mode: {audio_feat} not supported!')

    if task == -1 or task == 3:
        logging.info('====== Extract Images from Video ======')
        img_list = extract_imgs_from_video(video_file=video_file, ori_imgs_dir=ori_imgs_dir, resize_shape=resize_shape)

    if task == -1 or task == 4:
        img_list = img_list or extract_imgs_from_video(video_file=video_file, resize_shape=resize_shape)
        # logging.info('====== Detect Face Landmarks ======')
        # (deprecated caz face-alignment is too slow, use mediapipe only)
        # fa2d = FaceAlignmentLandmark(lms_type='2D')
        # fa2d_lms = fa2d.generate_face_alignment_lms(img_list=img_list)
        # fa2d_lms_mask = [True if lms is not None else False for lms in fa2d_lms]
        # fa3d = FaceAlignmentLandmark(lms_type='3D')
        # fa3d_lms = fa3d.generate_face_alignment_lms(img_list=img_list)

        logging.info('====== Detect Face Mesh & Track Pose ======')
        # Reference: https://github.com/Rassibassi/mediapipeFacegeometryPython
        from face_mesh.extract_face_mesh import MediapipeFaceMesh, transform_faceposes_to_cameraposes
        mp_mesh = MediapipeFaceMesh()
        face_mesh, face_poses = mp_mesh.extract_face_mesh(img_list=img_list, debug_freq=debug_freq, debug_dir=debug_dir)
        file_ops.json_dump(face_mesh, os.path.join(ori_imgs_dir, 'face_mesh.json'))
        file_ops.json_dump(face_poses, os.path.join(ori_imgs_dir, 'face_poses.json'))

        camera_poses = transform_faceposes_to_cameraposes(face_poses=face_poses)
        file_ops.json_dump(camera_poses, os.path.join(ori_imgs_dir, 'camera_poses.json'))
        del mp_mesh, face_mesh, face_poses, camera_poses

    if task == -1 or task == 5:
        img_list = img_list or extract_imgs_from_video(video_file=video_file, resize_shape=resize_shape)
        logging.info('====== Parse Face ======')
        from face_parsing.parse_face import FaceParser
        face_parser = FaceParser()
        parsed_images = face_parser.parse_faces(img_list=img_list,
                                                parsing_dir=parsing_dir,
                                                debug_freq=debug_freq,
                                                debug_dir=debug_dir)
        del face_parser, parsed_images

    if task == -1 or task == 6:
        logging.info('====== Extract Background ======')
        bc_img = extract_background(base_dir=base_dir)

    if task == -1 or task == 7:
        img_list = img_list or extract_imgs_from_video(video_file=video_file, resize_shape=resize_shape)
        logging.info('====== Extract torso images and gt_images ======')
        extract_torso_and_gt(base_dir=base_dir)

    if task == -1 or task == 8:
        logging.info('====== Extract Depth ======')
        from face_depth import DaganFaceDepth
        dagan_depth = DaganFaceDepth()
        face_depth = dagan_depth.extract_face_depth(img_list=img_list, debug=True, save_dir=debug_dir)
        for i, d in enumerate(face_depth):
            np.save(os.path.join(depth_dir, f'{i}_depth.npy'), d)

    if task == -1 or task == 9:
        logging.info('====== Save transforms.json ======')
        save_transform(base_dir=base_dir)

    logging.info('Done.')


if __name__ == '__main__':
    set_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='video file path')
    parser.add_argument('--task', type=int, help='which task to process', default=-1)
    parser.add_argument('--resize_shape', type=int, help='resize video to size*size', default=None)
    parser.add_argument('--audio_feat', type=str, choices=['hubert'], help='audio feature type', default='hubert')
    parser.add_argument('--debug_freq',
                        type=int,
                        help='frequency to store intermediate results for debugging',
                        default=0)

    opt = parser.parse_args()

    base_dir = os.path.dirname(opt.video_path)
    if opt.resize_shape is None:
        resize_shape = opt.resize_shape
    else:
        resize_shape = (int(opt.resize_shape), int(opt.resize_shape))

    # --- uncomment VizTracer to tackle each step ---
    # from viztracer import VizTracer
    # with VizTracer(output_file="viz.html") as tracer:
    # ----------------------------------------------
    process_video(
        video_file=opt.video_path,
        base_dir=base_dir,
        task=opt.task,
        resize_shape=resize_shape,
        audio_feat=opt.audio_feat,
        debug_freq=opt.debug_freq,
    )
