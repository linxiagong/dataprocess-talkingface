# Talking Face - Data Processing Pipeline 
This is a ready-to-use data processing pipeline for talking avatar videos.

todo: add the result video here


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
python3 -m pip install -r data_process/requirements.txt
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
-fd
```
For downstream training, here is the dataset (pytorch) for use: frames_dataset.py

## Implementation Details
### Audio Processing

### Face Landmarks and Mesh
Reference: Mediapipe
We found Mediapipe works better than landmarks

### Face Parsing
Reference: [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)
### Face Depth
Reference: [DaGAN](https://github.com/harlanhong/CVPR2022-DaGAN/tree/master)

## Reference
- ADNeRF
- RADNeRF