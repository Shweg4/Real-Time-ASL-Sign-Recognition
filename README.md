# Real-Time ASL Static Sign Recognition

Recognizes static American Sign Language (ASL) alphabet signs in real time from a webcam, video, or photo.

Instead of classifying raw images directly, the pipeline first extracts the 3D coordinates of the hand's joints (called landmarks), then classifies signs from those coordinates. This keeps the model small and fast, and makes it far less sensitive to background, lighting, or skin tone than a model trained on raw pixels would be.

The pipeline has three stages:

1. **Landmark extraction.** [Mediapipe](https://developers.google.com/mediapipe)'s hand landmarker detects 21 keypoints per hand (fingertips, knuckles, wrist, etc.), each with (x, y, z) coordinates, giving a 63-number description of the hand's shape for every image. Images are also horizontally flipped and re-labeled, so the model sees both left and right hands during training.
2. **Model training.** A classifier maps that 63-number vector to one of the alphabet classes. Two different model architectures are implemented and compared (see "Models" below).
3. **Real-time inference.** The trained model runs on a live webcam feed (or a video/photo file): Mediapipe detects the hand and its landmarks in each frame, the model predicts a sign, and the prediction is drawn on screen with a confidence score.

## Models

Two independent classifiers are trained on the same landmark data, using two different ways of thinking about what a "hand" is.

### 1. 1D-CNN + Transformer hybrid (`src/`, `notebooks/`)

This is the main model. The 63 landmark values are treated as a short 1D sequence and fed through:

- **1D convolutional layers first**, to pick up local patterns among landmarks that are numbered near each other (e.g. the three joints along one finger).
- **A Transformer block (self-attention) after that**, so the model can also relate landmarks that are far apart in the numbering but matter together for a given sign (e.g. the thumb tip and the pinky tip, which sit at opposite ends of the 21-point list but are often the two points that define a shape like "L" or "Y").

TensorFlow/Keras is used here. Training uses early stopping and learning-rate reduction on plateau, so it stops automatically once validation accuracy stops improving rather than always running the full epoch budget.

### 2. PointNet (`pointnet/`), an alternative approach

The CNN+Transformer model treats the 21 landmarks as if they were in a meaningful order (landmark #5 next to landmark #6, and so on), because that's how Mediapipe numbers them. But a hand isn't really a sequence, it's a set of points in 3D space with no natural order. PointNet is a model architecture designed for exactly that: point clouds.

It works by running the same small neural network independently on every point, then combining all 21 results with a single max-pooling step. Because max-pooling doesn't care what order its inputs arrive in, the model's prediction is unaffected by the order of the landmarks, matching how a hand actually is (a shape, not a sequence).

This version is implemented in PyTorch and is trained with 3D data augmentation (random rotation, scaling, and Gaussian noise on the point cloud) to make it robust to hand orientation and camera angle. It also includes a proper evaluation script that produces a confusion matrix and saves misclassified images for inspection, which the CNN+Transformer pipeline does not currently have.

Both models only cover *static* alphabet signs. Letters that require motion to sign correctly (`j` and `z`) and the digits are excluded, since a single still frame of landmarks can't capture motion.

## Repo layout

```
notebooks/
  ASL_Sign_Recognition.ipynb   # full CNN+Transformer pipeline in one notebook (e.g. for Colab)
src/
  video_to_frames.py           # optional: build a dataset from your own recorded videos
  extract_landmarks.py         # Part 1: dataset -> landmarks pickle
  model.py                     # CNN + Transformer architecture
  train.py                     # Part 2: train + evaluate the model
  infer.py                     # Part 3: real-time / video / photo inference
pointnet/
  ...                          # PointNet alternative, see "Models" above
```

Use the notebook if you want to run the whole thing end-to-end in one place (e.g. Colab, with GPU). Use the `src/` scripts if you want to run or reuse individual stages locally.

## Setup

```bash
pip install -r requirements.txt
wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

You need a folder of static ASL alphabet images arranged one subfolder per class (e.g. `A/`, `B/`, `C/`, ...), such as the [ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) or [ASL Alphabet Train](https://www.kaggle.com/datasets/ameythakur20/asl-alphabet-train?resource=download) on Kaggle. If you don't have one yet, `src/video_to_frames.py` can build it from your own recorded videos (see step 0 below).

## Usage

```bash
# 0. (Optional) Build an image dataset from your own recorded sign videos.
#    Input: a folder of per-class video subfolders, e.g. videos/A/*.mp4, videos/B/*.mp4
#    Output: a folder of per-class frame images, e.g. frames/A/1.jpg, frames/A/2.jpg
python src/video_to_frames.py --input-dir path/to/videos --output-dir path/to/frames --fps 15

# 1. Extract landmarks
python src/extract_landmarks.py --dataset-dir path/to/asl_alphabet_train --model-path hand_landmarker.task --output landmarks.pkl.gz

# 2. Train
python src/train.py --data landmarks.pkl.gz --model-out asl_model.keras --label-encoder-out label_encoder.pkl

# 3. Run inference
python src/infer.py --model asl_model.keras --label-encoder label_encoder.pkl --mode live
```

## PointNet quickstart

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

- Datasets and trained model artifacts are not committed to this repo (see `.gitignore`), regenerate them with the scripts above.
- Training uses early stopping and learning-rate reduction on plateau, so `--epochs` is an upper bound rather than a fixed run length.
