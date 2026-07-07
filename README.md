# Local Context Enhancement for Lightweight Real-Time UAV Object Detection on Edge Devices

## Abstract

This repository contains the implementation of **LECA-DETR**, a lightweight real-time object detector for UAV edge deployment. LECA-DETR is built on the Ultralytics RT-DETR pipeline and focuses on reducing model complexity while preserving small-object detection accuracy in aerial scenes.

The key contributions include:

1. **GhostNetV2 backbone adaptation**: replaces the original RT-DETR ResNet backbone with GhostNetV2 to reduce parameters and GFLOPs for edge-side inference.
2. **LECA (Local-Enhanced Context Attention)**: a lightweight dual-branch attention module that combines cross-shaped strip depthwise convolution and 1D channel interaction.
3. **Plug-in feature enhancement strategy**: inserts LECA after the P3/P4/P5 feature projections to keep the standard RT-DETR decoder input format unchanged.
4. **UAV edge deployment validation**: evaluates LECA-DETR on UAV datasets and reports TensorRT acceleration results on NVIDIA Jetson Orin NX.

## Environment Configuration

### Hardware Requirements

- NVIDIA GPU for training
- Recommended training GPU: RTX 3090 or above
- Edge deployment platform used in the paper: NVIDIA Jetson Orin NX
- CUDA-compatible graphics card

### Software Dependencies

The project is implemented based on the Ultralytics framework.

- Python 3.9+
- PyTorch
- CUDA
- Ultralytics
- timm
- OpenCV
- NumPy

### Conda Environment Configuration

```bash
# Create an environment
conda create -n lecadetr python=3.9

# Activate the environment
conda activate lecadetr

# Install PyTorch according to your CUDA version
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install core dependencies
pip install ultralytics timm opencv-python numpy
```

> Note: Please match the PyTorch and CUDA versions to your local GPU driver. The training setting in the paper uses an RTX 3090 GPU.

## Data Preparation

### VisDrone2019

LECA-DETR is mainly evaluated on the VisDrone2019 object detection dataset. Please download the dataset from the official VisDrone website and convert or organize it into the YOLO/Ultralytics detection format.

A typical dataset YAML can be configured as follows:

```yaml
# Ultralytics YOLO 🚀, AGPL-3.0 license
# VisDrone2019-DET dataset https://github.com/VisDrone/VisDrone-Dataset by Tianjin University
# Documentation: https://docs.ultralytics.com/datasets/detect/visdrone/
# Example usage: yolo train data=VisDrone.yaml
# parent
# ├── ultralytics
# └── datasets
#     └── VisDrone  ← downloads here (2.3 GB)


# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/VisDrone  # dataset root dir
train: VisDrone2019-DET-train/images  # train images (relative to 'path')  6471 images
val: VisDrone2019-DET-val/images  # val images (relative to 'path')  548 images
test: VisDrone2019-DET-test-dev/images  # test images (optional)  1610 images

# Classes
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor


# Download script/URL (optional) ---------------------------------------------------------------------------------------
download: |
  import os
  from pathlib import Path

  from ultralytics.utils.downloads import download

  def visdrone2yolo(dir):
      from PIL import Image
      from tqdm import tqdm

      def convert_box(size, box):
          # Convert VisDrone box to YOLO xywh box
          dw = 1. / size[0]
          dh = 1. / size[1]
          return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

      (dir / 'labels').mkdir(parents=True, exist_ok=True)  # make labels directory
      pbar = tqdm((dir / 'annotations').glob('*.txt'), desc=f'Converting {dir}')
      for f in pbar:
          img_size = Image.open((dir / 'images' / f.name).with_suffix('.jpg')).size
          lines = []
          with open(f, 'r') as file:  # read annotation.txt
              for row in [x.split(',') for x in file.read().strip().splitlines()]:
                  if row[4] == '0':  # VisDrone 'ignored regions' class 0
                      continue
                  cls = int(row[5]) - 1
                  box = convert_box(img_size, tuple(map(int, row[:4])))
                  lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
                  with open(str(f).replace(f'{os.sep}annotations{os.sep}', f'{os.sep}labels{os.sep}'), 'w') as fl:
                      fl.writelines(lines)  # write label.txt


  # Download
  dir = Path(yaml['path'])  # dataset root dir
  urls = ['https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-train.zip',
          'https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-val.zip',
          'https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-test-dev.zip',
          'https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-test-challenge.zip']
  download(urls, dir=dir, curl=True, threads=4)

  # Convert
  for d in 'VisDrone2019-DET-train', 'VisDrone2019-DET-val', 'VisDrone2019-DET-test-dev':
      visdrone2yolo(dir / d)  # convert VisDrone annotations to YOLO labels
```

Update the `data` path in `train.py` before training:

```python
model.train(data=r'VisDrone.yaml', imgsz=1024, epochs=400, batch=6)
```
[Dataset Download](https://github.com/VisDrone/VisDrone-Dataset)
### CODrone

The paper also reports results on CODrone. Since CODrone annotations are rotated boxes, the experiments convert each four-point rotated box into the minimum enclosing horizontal bounding box and then export normalized YOLO-format labels.

#### Preparation

CODrone annotations are provided as VOC-style rotated boxes. For the horizontal-box detection experiments, convert each four-point OBB annotation into its minimum enclosing horizontal bounding box and export normalized YOLO labels.

Place the original CODrone dataset at `D:\temp\CODrone` with this structure:

```text
D:\temp\CODrone
|-- train
|   |-- images
|   `-- labels
|-- val
|   |-- images
|   `-- labels
`-- test
    |-- images
    `-- labels
```

Run the converter:

```bash
python prepare_codrone.py --source D:\temp\CODrone --output D:\datasets\CODrone --clean
```

The script writes the Ultralytics-compatible dataset to:

```text
D:\datasets\CODrone
|-- classes.txt
|-- images
|   |-- train
|   |-- val
|   `-- test
`-- labels
    |-- train
    |-- val
    `-- test
```

It also updates `ultralytics/cfg/datasets/CODrone.yaml` to use `D:/datasets/CODrone`. 
Update the `data` path in `train.py` before training:

```python
model.train(data=r'CODrone.yaml', imgsz=1024, epochs=400, batch=6)

[Dataset Download](https://github.com/AHideoKuzeA/CODrone-A-Comprehensive-Oriented-Object-Detection-benchmark-for-UAV)

## Usage

### 1. Integrate Custom Modules

Copy the custom modules into the Ultralytics environment or project where model YAML parsing can import them:

```text
Addmodules/GhostNetV2.py
Addmodules/LECA.py
```

Then register/import `Ghostnetv2` and `LECA` in the corresponding Ultralytics module registry or model parsing file according to your local Ultralytics version.

### 2. Copy Model YAML Files

Copy the model configuration files to the Ultralytics model configuration directory or keep them in the working directory:

```text
yaml/rtdetr-Ghostnet.yaml
yaml/rtdetr-GhostnetLECA.yaml
```

### 3. Train

```bash
python train.py
```

The default training script uses:

```python
model = RTDETR('rtdetr-GhostnetLECA.yaml')
model.train(
    data=r'VisDrone.yaml',
    cache=False,
    imgsz=1024,
    epochs=400,
    batch=6,
    workers=0,
    device='0',
    project='visdrone_rtdetr',
    name='exp',
)
```

### 4. Train Other Variants

To train the GhostNetV2-only baseline, change the model YAML path:

```python
model = RTDETR('rtdetr-Ghostnet.yaml')
```

To train LECA-DETR, use:

```python
model = RTDETR('rtdetr-GhostnetLECA.yaml')
```

## Core Algorithms

### 1. GhostNetV2 Backbone

GhostNetV2 is used to replace the default ResNet backbone in RT-DETR. It generates redundant feature maps through cheap operations and introduces lightweight Decoupled Fully Connected attention in deeper blocks.

In LECA-DETR, GhostNetV2 is used only as a multi-scale feature extractor. The detection head receives three feature levels corresponding to P3, P4, and P5, which are then projected to a unified 256-channel hidden dimension.

**Key Features:**

- Lightweight feature generation through Ghost modules
- Lower parameter count and computational cost than ResNet50
- Multi-scale outputs compatible with RT-DETR hybrid encoder and decoder

### 2. Local-Enhanced Context Attention (LECA)

LECA is a lightweight attention module designed for UAV small-object detection. It contains a spatial branch and a channel branch.

**Spatial Branch:**

```text
S_h = DWConv(1 x K)(X)
S_w = DWConv(K x 1)(X)
W_spatial = Sigmoid(S_h + S_w)
```

The cross-shaped strip depthwise convolution enhances horizontal and vertical local edge responses while avoiding the heavier cost of square convolution.

**Channel Branch:**

```text
W_channel = Sigmoid(Conv1D(GAP(X * W_spatial)))
Y = (X * W_spatial) * W_channel
```

The channel branch performs lightweight cross-channel interaction with a 1D convolution, avoiding fully connected layers and channel reduction.

**Key Features:**

- Strip depthwise convolution with default kernel size 5
- 1D convolution for channel interaction
- Channel-preserving input/output format
- Approximately 0.01M additional parameters for three inserted LECA modules

### 3. LECA Insertion Strategy

LECA is inserted after the `1 x 1` projection convolutions of P3, P4, and P5. At this stage, all features have been aligned to 256 channels.

This design keeps:

- Spatial resolutions unchanged
- Channel dimensions unchanged
- RT-DETR decoder input format unchanged
- Plug-in compatibility with the original feature fusion path

## Main Results

### VisDrone2019 Results

| Model | Input Resolution | mAP50 (%) | mAP50-95 (%) | Params (M) | GFLOPs |
| --- | --- | ---: | ---: | ---: | ---: |
| RT-DETR + ResNet50 | 1024 x 1024 | 51.19 | 31.70 | 41.96 | 125.7 |
| RT-DETR + GhostNetV2 | 1024 x 1024 | 48.15 | 29.65 | 12.39 | 25.9 |
| RT-DETR + GhostNetV2 + LECA | 1024 x 1024 | 50.55 | 31.81 | 12.40 | 26.0 |

Compared with the RT-DETR ResNet50 baseline, LECA-DETR reduces parameters by **70.4%** and GFLOPs by **79.3%**, while improving mAP50-95 by **0.11 percentage points**.

### CODrone Results

| Model | Input Resolution | mAP50 (%) | mAP50-95 (%) | Params (M) | GFLOPs |
| --- | --- | ---: | ---: | ---: | ---: |
| RT-DETR + ResNet50 | 1024 x 1024 | 32.24 | 15.92 | 41.96 | 125.7 |
| RT-DETR + GhostNetV2 | 1024 x 1024 | 35.41 | 18.81 | 12.39 | 25.9 |
| RT-DETR + GhostNetV2 + LECA | 1024 x 1024 | 35.52 | 19.10 | 12.40 | 26.0 |

### Jetson Orin NX Deployment

| Model | TensorRT | Avg Latency (ms) | FPS |
| --- | --- | ---: | ---: |
| RT-DETR + ResNet50 | No | 1975.56 | 0.5 |
| RT-DETR + GhostNetV2 | No | 655.89 | 1.5 |
| RT-DETR + GhostNetV2 + LECA | No | 694.09 | 1.4 |
| RT-DETR + ResNet50 | Yes | 48.75 | 20.5 |
| RT-DETR + GhostNetV2 | Yes | 33.17 | 30.1 |
| RT-DETR + GhostNetV2 + LECA | Yes | 36.65 | 27.3 |

With TensorRT acceleration, LECA-DETR reaches **36.65 ms** latency and **27.3 FPS** on NVIDIA Jetson Orin NX.

## Pretrained Models
- [LECA-DETR VisDrone 1024](https://drive.google.com/file/d/1d-df_MC-WUMtcO-kutVka6_bFlptUm4y/view?usp=sharing)
- [LECA-DETR CODrone 1024](https://drive.google.com/file/d/1iUknUaMXOD1tr2oAq0i8fYDxInxcM1GC/view?usp=sharing)
## Citation

If this project is helpful for your research, please cite our paper:

```bibtex

```

## Acknowledgments

We thank the following projects and datasets for supporting this research:

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the RT-DETR implementation framework
- [RT-DETR](https://github.com/lyuwenyu/RT-DETR) for real-time end-to-end detection research
- [GhostNetV2](https://github.com/huawei-noah/Efficient-AI-Backbones) for lightweight backbone design
- [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) for UAV object detection benchmarks
- [CODrone](https://github.com/AHideoKuzeA/CODrone-A-Comprehensive-Oriented-Object-Detection-benchmark-for-UAV) for UAV object detection benchmarks
