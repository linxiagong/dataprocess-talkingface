"""Main process for data processing.
Every Speaker into one dataset.
"""
import argparse
import logging
import os
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
import torch
import cv2
import numpy as np

# import skimage
from tqdm import trange

from data_process.face_landmark import FaceAlignmentLandmark
from data_process.face_depth import DaganFaceDepth
from data_process.face_mesh.extract_face_mesh import MediapipeFaceMesh
# from data_process.face_mesh import MediapipeFaceMesh
# from data_process.face_parsing import FaceParser
from data_process.face_parsing.parse_face import FaceParser
from data_process.audio_process.hubert import HubertFeatureExtractor
from data_process.audio_process import extract_wav_from_video

# from face_pose import extract_3ddfa_face_pose
from viztracer import VizTracer
# from process_audio import extract_audio_feature, extract_wav_from_video
from dataprocess_ray.ray_utils import ray_shutdown, ray_init, ray_run

NOCOLOR = "\033[0m"
YELLOW = "\033[1;33m"
BLUE = "\033[1;34m"

def extract_imgs_from_video(video_file: str, resize_shape: tuple = None) -> list:
    """Extract original images for a video file, return a list.
    """
    frame_list = list()
    cap = cv2.VideoCapture(str(video_file))
    while (True):
        _, frame = cap.read()
        if frame is None:
            break
        if isinstance(resize_shape, tuple):
            frame = cv2.resize(frame, resize_shape)
            # frame = skimage.transform.resize(frame, resize_shape, preserve_range=True)
        # cv2.imwrite(os.path.join(ori_imgs_dir, str(frame_num) + '.jpg'), frame)
        frame_list.append(frame.astype(np.uint8))
    cap.release()
    return frame_list

def process_speaker_videos(
    speaker_dir: Path,  # <dataset_folder>/<speaker_id>/
    output_dir: str,
    resize_shape: tuple = None,
    # audio_feat_type: str = "deepspeech",
    skip_exist: bool = True,
    debug: bool = False,
):
    """Process videos of a speaker.
    """
    # init timing and flags
    t_process_start = time.time()
    is_success = False

    # set output path
    speaker_id = Path(speaker_dir).name
    working_dir = os.path.join(output_dir, speaker_id)   # to store all output data about this speaker
    os.makedirs(working_dir, exist_ok=True)
    print(f'speaker_id={speaker_id}| working_dir={working_dir}')

    wav_dir = os.path.join(working_dir, 'wav')
    os.makedirs(wav_dir, exist_ok=True)
    
    if resize_shape is not None:
        dataset_dir = os.path.join(working_dir, f'dataset_{resize_shape[0]}')
    else:
        dataset_dir = os.path.join(working_dir, f'dataset_orig')
    print(f'speaker_id={speaker_id}| dataset_dir={dataset_dir}')
    os.makedirs(dataset_dir, exist_ok=True)

    # extra info record
    try:
        extra = np.load(dataset_dir + '/extra_info.npy', allow_pickle='TRUE').item()
    except:
        extra = defaultdict(dict)

    # start to process videos
    video_files = sorted(Path(speaker_dir).glob('**/*.mp4'), key=lambda x: x.relative_to(speaker_dir))

    extra["video_files"] = video_files

    # --- uncomment VizTracer to tackle each step ---
    # with VizTracer(output_file="viz.html") as tracer:
    # ----------------------------------------------
    try:
        # prepare models, load models only once for images
        print('Prepare models...')
        # fa2d = FaceAlignmentLandmark(lms_type='2D')
        # fa3d = FaceAlignmentLandmark(lms_type='3D')
        mp_mesh = MediapipeFaceMesh()
        face_parser = FaceParser()
        dagan_depth = DaganFaceDepth()
        hubert_extractor = HubertFeatureExtractor()

        print('Start video process...')
        total_frame_num = 0
        for video_idx, video_file in enumerate(video_files):
            t = time.time()

            video_rela_path = str(video_file.relative_to(speaker_dir))[:-4]
            video_id = speaker_id + '/' + video_rela_path
            
            dataset_path = dataset_dir + '/' + video_id.replace('/', '#')
            if skip_exist and dataset_exist(dataset_path):
                continue
            dataset_builder = IndexedDatasetBuilder(dataset_path)
            print(f'{YELLOW}{video_id}{NOCOLOR}| start...')
            # Set Debug Folder
            debug_dir = os.path.join(working_dir, 'debug', video_rela_path.replace('/', '#'))
            os.makedirs(debug_dir, exist_ok=True)

            # Open the MP4 file
            video = cv2.VideoCapture(str(video_file))
            # Get the total number of frames in the video
            video_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_rate = int(video.get(cv2.CAP_PROP_FPS))
            extra[video_id]['fps_rate'] = fps_rate
            print(f'{YELLOW}{video_id}{NOCOLOR}| frame_count={video_len}, fps_rate={fps_rate}')
    
            # Step 0: extract wav & deepspeech feature, better run in terminal to parallel with
            # below commands since this may take a few minutes
            wav_file = os.path.join(wav_dir, video_rela_path.replace('/', '#') + '.wav')
            extract_wav_from_video(video_file=video_file, wav_file=wav_file)
            t1 = time.time()
            hubert_features = hubert_extractor.extract_hubert_features(wav_file, target_len=video_len, target_fps=fps_rate)
            t2 = time.time()
            print(f'{YELLOW}{video_id}{NOCOLOR}| video_len={video_len}/{total_frame_num}| hubert done. Cost {t2-t1:.4f}s, avg={(t2-t1)/video_len:.4f}s')
    
            # Step 1: extract images
            img_list = extract_imgs_from_video(video_file=video_file, resize_shape=resize_shape)
            video_len = len(img_list)
            extra[video_id]['video_len'] = video_len
            print(f'{YELLOW}{video_id}{NOCOLOR}| extract {video_len}/{total_frame_num} frames.')

            # Step 2: detect landmarks (deprecated caz face-alignment is too slow, use mediapipe only)
            # t1 = time.time()
            # fa2d_lms = fa2d.generate_face_alignment_lms(img_list=img_list)
            # fa2d_lms_mask = [True if lms is not None else False for lms in fa2d_lms]
            # extra[video_id]['fa2d_lms_mask'] = fa2d_lms_mask
            # t2 = time.time()
            # print(f'{YELLOW}{video_id}{NOCOLOR}| video_len={video_len}|  {sum(fa2d_lms_mask)} 2D landmark done. Cost {t2-t1:.4f}s, avg={(t2-t1)/video_len:.4f}s')

            # t1 = time.time()
            # fa3d_lms = fa3d.generate_face_alignment_lms(img_list=img_list)
            # extra[video_id]['fa3d_lms_mask'] = [True if lms is not None else False for lms in fa2d_lms]
            # t2 = time.time()
            # print(f'{YELLOW}{video_id}{NOCOLOR}| video_len={video_len}|  3D landmark done. Cost{t2-t1:.4f} s, avg={(t2-t1)/video_len:.4f}s')
        
            # Step 3: extract face mesh & pose
            # Reference: https://github.com/Rassibassi/mediapipeFacegeometryPython
            t1 = time.time()
            face_mesh, faces_poses = mp_mesh.extract_face_mesh(img_list=img_list, debug=debug, save_dir=debug_dir)
            extra[video_id]['face_mesh_mask'] = [True if mesh is not None else False for mesh in face_mesh]
            # Step 3.1: pose
            # Reference: https://github.com/cleardusk/3DDFA
            # faces_poses = extract_3ddfa_face_pose(img_list=img_list)
            # Mediapipe works better for face pose
            t2 = time.time()
            print(f'{YELLOW}{video_id}{NOCOLOR}| video_len={video_len}/{total_frame_num}|  mesh and pose done. Cost {t2-t1:.4f}s, avg={(t2-t1)/video_len:.4f}s')
        
            # Step 4: face parsing
            # https://github.com/zllrunning/face-parsing.PyTorch
            t1 = time.time()
            parsed_images = face_parser.parse_faces(img_list=img_list, debug=debug, save_dir=debug_dir)
            t2 = time.time()
            print(f'{YELLOW}{video_id}{NOCOLOR}| video_len={video_len}/{total_frame_num}|  parse face done. Cost {t2-t1:.4f}s, avg={(t2-t1)/video_len:.4f}s')
            
            # Step 5: depth
            # https://github.com/harlanhong/Face-Depth-Network
            t1 = time.time()
            face_depth = dagan_depth.extract_face_depth(img_list=img_list, debug=True, save_dir=debug_dir)
            t2 = time.time()
            print(f'{YELLOW}{video_id}{NOCOLOR}| video_len={video_len}/{total_frame_num}|  face depth done. Cost {t2-t1:.4f}s, avg={(t2-t1)/video_len:.4f}s')

            # Save to dataset
            for i in trange(video_len):
                frame = {
                    "speaker_id": str(speaker_id),
                    "video_id": video_id,
                    "image_id": int(i),
                    "image": img_list[i].astype(np.int8),  # resized
                    # "focal_len": None,
                    "pose": faces_poses[i],
                    # # "fa2d_lms": fa2d_lms[i],  # face alignment 2D landmarks
                    # # "fa3d_lms": fa3d_lms[i],
                    "face_mesh": face_mesh[i],
                    "image_parsed": parsed_images[i].astype(np.int8),
                    "depth": face_depth[i],
                    # "audio_deepspeech": None,
                    "audio_hubert": hubert_features[i],
                }
                dataset_builder.add_item(frame)
                total_frame_num += 1
            dataset_builder.finalize()
            print(f' {YELLOW}{video_id}{NOCOLOR}| video_len={video_len}/{total_frame_num}|  Done. Time cost {time.time()- t:.4f}s.')
        np.save(dataset_dir + '/extra_info.npy', extra)

        print(f'speaker_id={speaker_id} done. Time cost {time.time()-t_process_start:.4f}s.')
        is_success = True
    except Exception as e:
        print(f'ERROR {speaker_id}: {e}')
    return [(speaker_id, is_success)]
    

def set_logging(save_to_file:bool=True):
    # Basic logging set
    handlers = [logging.StreamHandler()]
    if save_to_file:
        logfile = f'./log/data_process/main_process.log'
        os.makedirs(f'./log/data_process/', exist_ok=True)
        handlers += [logging.FileHandler(logfile, 'a')]
    logging.basicConfig(level=logging.INFO,
                        format=f'%(asctime)s - %(filename)s:%(lineno)s: ' + f'{BLUE}%(message)s{NOCOLOR}',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=handlers)
    

if __name__ == '__main__':
    set_logging()

    logging.info('NEW EXPERIMENT STARTS...')
    parser = argparse.ArgumentParser()
    parser.add_argument('--speaker_dir', type=str, default='')
    parser.add_argument('--all_speaker_dir', type=str, default='')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--resize_shape', type=int, help='resize video to size*size', default=None)
    # parser.add_argument('--cpu_cnt', type=int, help='cpu cnt for process')
    parser.add_argument('--debug', action='store_true', help='use debug mode')
    args = parser.parse_args()

    t_start = time.time()

    # parse arguments
    assert (args.speaker_dir or args.all_speaker_dir), "Specify either --all_speaker_dir or --speaker_dir"

    if args.resize_shape is not None:
        resize_shape = (args.resize_shape, args.resize_shape)
        logging.info(f'Images are resized to {resize_shape}')
    else:
        resize_shape = None
        logging.info(f'Do not resize images.')
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    # process videos
    # if os.path.isdir(args.all_speaker_dir):
    #     logging.info(f'Iterate over all_speaker_dir={args.all_speaker_dir}')
    #     import multiprocessing
    #     cpu_cnt = args.cpu_cnt or int(multiprocessing.cpu_count() * 0.9)
    #     p = multiprocessing.Pool(cpu_cnt)
    #     for speaker_dir in Path(args.all_speaker_dir).iterdir():
    #         if speaker_dir.is_dir():
    #             p.apply_async(
    #                 partial(
    #                     process_speaker_videos,
    #                     speaker_dir=speaker_dir,
    #                     output_dir=args.output_dir,
    #                     resize_shape=resize_shape,
    #                     skip_exist=False,
    #                     debug=args.debug,
    #                 ))
    if os.path.isdir(args.speaker_dir):
        logging.info(f'Process speaker_dir={args.speaker_dir}')
        process_speaker_videos(
            speaker_dir=args.speaker_dir,
            output_dir=args.output_dir,
            resize_shape=resize_shape,
            skip_exist=False,
            debug=args.debug,
        )
        exit()

    
    if os.path.isdir(args.all_speaker_dir):
        logging.info(f'Iterate over all_speaker_dir={args.all_speaker_dir}')
        items = [(speaker_dir,) for speaker_dir in Path(args.all_speaker_dir).iterdir() if speaker_dir.is_dir()]
        items = sorted(items, key=lambda x:x[0].name)[:1000]
        
        # if dataset exist, skip
        # def skip_processed_id(dirs_list:list):
        #     unprocessed_dirs = []
        #     for speaker_dir in dirs_list:
        #         speaker_id = Path(speaker_dir[0]).name
        #         working_dir = os.path.join(args.output_dir, speaker_id)
        #         if resize_shape is not None:
        #             output_path = os.path.join(working_dir, f'{speaker_id}_{resize_shape[0]}')
        #         else:
        #             output_path = os.path.join(working_dir, f'{speaker_id}')
        #         if dataset_exist(output_path):
        #             continue
        #             print(f'speaker_id={speaker_id}| dataset exists, skip...')
        #         else:
        #             unprocessed_dirs.append(speaker_dir)
        #     return unprocessed_dirs

        # items = skip_processed_id(items)
        logging.info(f'process ids: {[x[0].name for x in items]}. len={len(items)}')

        # --- Ray ---
        results_success = []
        results_fail = []
        ray_init()
        chunk_worker_kwargs=dict(output_dir=args.output_dir,
                            resize_shape=resize_shape,
                            skip_exist=True,
                            debug=args.debug,)
        ray_options = {'memory': 2.5 * 1024 * 1024 * 1024}
        for i, res in ray_run(process_speaker_videos, items, chunk_size=1, chunk_worker_kwargs=chunk_worker_kwargs, ray_options=ray_options):
            if res is not None:
                speaker_id, success = res
                if success:
                    results_success.append(speaker_id)
                else:
                    results_fail.append(speaker_id)
        ray_shutdown()
        logging.info(f'Done. Time cost {time.time()-t_start:.4f}s')
        results_success.sort()
        logging.info(f'{len(results_success)} succeed: {results_success}')
        logging.info(f'{len(results_fail)} fail: {results_fail}')
