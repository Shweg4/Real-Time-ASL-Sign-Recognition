"""
Part 1: Extract hand landmarks from the ASL alphabet image dataset using
Mediapipe, augment with horizontally flipped images, and save the result
as a compressed pickle for training.
"""
import argparse
import os

import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageOps

BORDER_SIZE = 100


def build_detector(model_path: str) -> vision.HandLandmarker:
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    return vision.HandLandmarker.create_from_options(options)


def extract_landmarks(detector: vision.HandLandmarker, image_path: str):
    source_img = Image.open(image_path)
    bordered = np.array(ImageOps.expand(source_img, border=BORDER_SIZE, fill="black"))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=bordered)

    detection_result = detector.detect(mp_image)
    if not detection_result.hand_landmarks:
        return None

    landmarks = [[lm.x, lm.y, lm.z] for lm in detection_result.hand_landmarks[0]]
    return source_img, landmarks


def build_dataset(directory_path: str, model_path: str) -> pd.DataFrame:
    detector = build_detector(model_path)
    rows = []
    skipped = 0

    for folder in os.listdir(directory_path):
        folder_path = os.path.join(directory_path, folder)
        if not os.path.isdir(folder_path):
            continue

        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if not os.path.isfile(item_path):
                continue

            result = extract_landmarks(detector, item_path)
            if result is None:
                skipped += 1
                continue

            source_img, landmarks = result
            rows.append({"class": folder, "image": np.array(source_img), "landmarks": landmarks})

            flipped_image = source_img.transpose(Image.FLIP_LEFT_RIGHT)
            flipped_landmarks = [[1.0 - lm[0], lm[1], lm[2]] for lm in landmarks]
            rows.append({"class": folder, "image": np.array(flipped_image), "landmarks": flipped_landmarks})

    print(f"images:{len(rows)}  skipped:{skipped}")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, help="Path to asl_alphabet_train folder")
    parser.add_argument("--model-path", default="hand_landmarker.task", help="Path to the Mediapipe hand landmarker model")
    parser.add_argument("--output", default="hand_landmarks.pkl.gz", help="Output path for the compressed landmarks dataset")
    args = parser.parse_args()

    df = build_dataset(args.dataset_dir, args.model_path)
    df.to_pickle(args.output)
    print(f"Saved {len(df)} rows to {args.output} ({os.path.getsize(args.output) / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
