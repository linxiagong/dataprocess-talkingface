# dataprocess-talkingface
This is a ready-to-use data processing pipeline for talking avatar videos.

todo: add the result video here

## Quick Start
&#9312; Install dependency
```
python3 -m pip install -r data_process/requirements.txt
```

**&#9313; Download necessary models:**

Download Depth Checkpoints of [DaGAN](https://github.com/harlanhong/CVPR2022-DaGAN#pre-trained-checkpoint), put the folder `depth_face_model_Voxceleb2_10w` under folder `data_process/face_depth/`.
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
**&#9314; Usage:**
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
Mediapipe

### Face Parsing
Reference: 
### Face Depth
Reference: [DaGAN](https://github.com/harlanhong/CVPR2022-DaGAN/tree/master)
