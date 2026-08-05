# Real-Time ASL Static Sign Recognition

Recognizes static American Sign Language (ASL) alphabet signs in real time from a webcam, video, or photo.

The pipeline has three stages:

1. **Landmark extraction** — [Mediapipe](https://developers.google.com/mediapipe) detects 21 hand landmarks (x, y, z) per image in the [ASL alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet), with horizontal-flip augmentation so the model generalizes to both hands.
2. **Model training** — a hybrid 1D-CNN + Transformer classifies the 63-dimensional landmark vector (21 points × 3 coordinates) into a sign class.
3. **Real-time inference** — the trained model runs on a live webcam feed (or a video/photo file), drawing a bounding box and predicted label over the detected hand.

## Repo layout

```
notebooks/
  ASL_Sign_Recognition.ipynb   # full CNN+Transformer pipeline in one notebook (e.g. for Colab)
src/
  extract_landmarks.py         # Part 1: dataset -> landmarks pickle
  model.py                     # CNN + Transformer architecture
  train.py                     # Part 2: train + evaluate the model
  infer.py                     # Part 3: real-time / video / photo inference
pointnet/
  ...                          # alternative approach, see below
```

Use the notebook if you want to run the whole thing end-to-end in one place (e.g. Colab, with GPU). Use the `src/` scripts if you want to run or reuse individual stages locally.

## Setup

```bash
pip install -r requirements.txt
wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

Download the [ASL alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) and point `--dataset-dir` at the `asl_alphabet_train/asl_alphabet_train` folder.

## Usage

```bash
# 1. Extract landmarks
python src/extract_landmarks.py --dataset-dir path/to/asl_alphabet_train --model-path hand_landmarker.task --output landmarks.pkl.gz

# 2. Train
python src/train.py --data landmarks.pkl.gz --model-out asl_model.keras --label-encoder-out label_encoder.pkl

# 3. Run inference
python src/infer.py --model asl_model.keras --label-encoder label_encoder.pkl --mode live
```

## Alternative approach: PointNet (`pointnet/`)

A second, independent modeling approach: instead of flattening the 21 landmarks into a 1D vector for the CNN+Transformer, `pointnet/` treats them as an unordered 3D point cloud and classifies them with a [PointNet](https://arxiv.org/abs/1612.00593)-style architecture (PyTorch), trained with 3D rotation/scale/Gaussian-noise augmentation. It also includes a proper evaluation script that produces a confusion matrix and dumps misclassified samples for inspection.

It only covers the *static* alphabet signs — `j` and `z` (which require motion) and digits are excluded, since this pipeline classifies single still frames.

```bash
pip install -r pointnet/requirements.txt
cd pointnet

# 1. Build the augmented point-cloud dataset from images (paths configured in config.yaml)
python create_points_data.py

# 2. Train
python train.py

# 3. Evaluate: accuracy, confusion matrix, misclassified-sample dump
python predict.py
```

Edit `pointnet/config.yaml` to point at your dataset, model, and output directories.

## Notes

- Datasets and trained model artifacts are not committed to this repo (see `.gitignore`) — regenerate them with the scripts above.
- Training uses early stopping and LR reduction on plateau, so `--epochs` is an upper bound rather than a fixed run length.
