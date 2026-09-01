# BENCHMARKING ROBUSTNESS OF SELF-SUPERVISED LEARNING ACROSS DIVERSE DOWNSTREAM TASKS

This is the official repository of the Master's Thesis of title Benchmarking Robustness of Self-Supervised Learning Across Diverse Downstream Tasks.

## Setup

The setup consists of 1) python environment; 2) pre-trained SSL encoders; 3) training and evaluation data.

### Environment

To set the environment up, you need to first install [uv](https://docs.astral.sh/uv/), a light-weight Python manager. On Linux/macOS it's as simple as `curl -LsSf https://astral.sh/uv/install.sh | sh`. Once installed, run `uv sync`, and `source .venv/bin/activate` to activate the new environment.

### SSL Encoders

To download the models, run `bash src/models/download.sh`.

### Data

To download segmentation and depth estimation data, download the samples from the respective sources ([ADE20k](https://ade20k.csail.mit.edu/), [Cityscapes](https://www.cityscapes-dataset.com/), [PASCAL VOC 2012](https://huggingface.co/datasets/nateraw/pascal-voc-2012), [NYU-Depth-v2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)) and the scripts in `src/data/parse_datasets` for pre-processing.

## Codebase Overview

The codebase implements the following: 1) adversarial attacks; 2) data handling; 3) evaluation (clean and adversarial); 4) SSL encoders and downstream linear heads; 5) downstream model training.

### Adversarial examples

`src/attack` contains implementation of all attacks evaluated in the thesis. `classification_pgd.py` implements the task-specific classification PGD, `depthpgd.py` implements DepthPGD, `embedattack.py`---EmbedAttack, and `segpgd.py`---SegPGD.

### Data

This module implements useful data handling functionality. Specifically: `parse_datasets` has scripts used to pre-process raw segmentation and depth estimation data, `loader.py` provides data-loading utilities for training and evaluation, and `local_datasets.py` defines classes of all datasets used in the thesis.

### Evaluation

Here are the utility functions for evaluating adversarial attacks and clean performance of the models. In `run.py` the pipelines for evaluating each of the downstream tasks under EmbedAttack and task-specific attacks is implemented. `adversarial.py` implements evaluation loops for specific instances of attacks, models, and dataset (under adversarial attacks), and `clean.py`---evaluation loops on clean images. 

### Models

This module implements: 1) ViT backbone in `backbones` folder; 2) downstream heads for each task (`downstream_models`); 3) SSL encoders `ssl_models`. `download.sh` allows easy download of the pre-trained DINOv1 and DINOv2 models, used in the thesis.

### Train

`trainer.py` implements training loops for all three downstream task types.

## Experiments

The supported experiments by the codebase are: 1) adversarial attacks; 2) downstream training. The configuration of each of the experiment type can be found in `conf/action`. To run either training or adversarial attack evaluation, execute:

```
python3 -m src.main +action=downstream_training +task=classification +ssl_model=dino_v1_vitb16
python3 -m src.main +action=adversarial_attack +task=classification +ssl_model=dino_v1_vitb16
```

### Tasks

The configuration files for three tasks are in `conf/task`. The available task types are: `classification`, `depth_estimation`, and `segmentation`. The dataset can be set by adding `task.dataset=[DATASET NAME]` in the command, for example:

```
python3 -m src.main +action=downstream_training +task=classification +ssl_model=dino_v1_vitb16 task.dataset=cifar100
```

### Models

The configuration files for the three SSL encoders are in `conf/ssl_model`. The available models are: `dino_v1_vitb16` (DINO-v1 ViT-B/16), `dino_v2_vitb14_distilled` (DINO-v2 ViT-B/14), and  `dino_v2_vits14_distilled` (DINO-v2 ViT-S/14).
