# Talking Face - Data Processing Pipeline 
This is a ready-to-use data processing pipeline for talking avatar videos.

https://user-images.githubusercontent.com/16673393/239467751-e8f13542-ca3c-433f-831d-6fa748d056f0.mp4


| This pipeline include: | By default, set `task=-1` to run all subtasks|
| :--- | :--- |
| task == 1 | Extract audio from video, saved as `audio.wav` |
| task == 2 | Extract audio features <br>Supported feature type: \[*hubert*\] |
| task == 3 | Extract original images from video |
| task == 4 | &#9312; Extract Face Mesh (substitute to landmarks) <br>&#9313; Track Face poses and camera poses |
| task == 5 | Parse Face by semantics |
| task == 6 | Calculate background images from video, based on semantics |
| task == 7 | Extract torso images, based on semantics |
| task == 8 | Predict depth for each frame of image |
| task == 9 | Split the dataset into train & eval sets <br>pytorch dataset file provided. |


## Quick Start
&#9312; Install dependency
```
$ python3 -m pip install -r requirements.txt
$ sudo apt-get install ffmpeg
```

**&#9313; Download necessary models:**

Download Depth Checkpoints of [DaGAN](https://github.com/harlanhong/CVPR2022-DaGAN#pre-trained-checkpoint), put the folder `depth_face_model_Voxceleb2_10w` under folder `data_process/face_depth/`, like this:
```
└── data_process
    └── face_depth
        └── depth_face_model_Voxceleb2_10w
            ├── depth.pth
            └── encoder.pth
        ├── extract_face_depth.py
        └── ...
    ├── face_mesh
    └── ...
```
**&#9314; Run the processing:**
```python
python data_process/porcess_video.py  
```
By default, the model uses GPU if available. If you want it not to use GPU, set `CUDA_VISIBLE_DEVICES=""`.

### Pytorch Dataset for use after data processing
After the data processing, the data is in a structure of:
```bash
└── base_dir
    ├── video.mp4
    ├── audio.wav
    └── ori_imgs
        ├── i.jpg
        ├── ...
        ├── camera_poses.json
        ├── face_mesh.json
        ├── face_poses.json
        └── nodes_points.json
    └── parsing
        ├── i.npy
        └── ...
    ├── bc.jpg
    └── gt_imgs
        ├── i.jpg
        └── ...
    └── torso_imgs
        ├── i.png
        └── ...
    └── depth
        ├── i.npy
        └── ...
    ├── transforms_train.json
    └── transforms_val.json


```
*TODO* a dataset.py for late use

## Implementation Details
### Audio Processing
Reference: HuBert
*TODO* Support more features

### Face Landmarks and Mesh
Reference: Mediapipe Face Landmark \[[Doc](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)\] \[[Github](https://github.com/google/mediapipe/tree/master)\]

`Mediapipe` works better than `face-alignment`.

### Face Parsing
Reference: [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)
### Face Depth
Reference: [DaGAN](https://github.com/harlanhong/CVPR2022-DaGAN/tree/master)

## Reference
- AD-NeRF [\[Project Link\]](https://yudongguo.github.io/ADNeRF/)
- RAD-NeRF [\[Project Link\]](https://github.com/ashawkey/RAD-NeRF)