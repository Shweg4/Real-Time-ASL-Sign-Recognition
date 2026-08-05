"""Build a training set from recorded sign videos by extracting frames at a target FPS.

Converts a folder of per-class video subfolders (e.g. Data/B/*.mp4, Data/C/*.mp4) into
a folder of per-class frame images (e.g. Data/B/1.jpg, Data/B/2.jpg, ...), suitable as
input to extract_landmarks.py.
"""
import argparse
import os

import cv2

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def video_to_frames(input_dir: str, output_dir: str, fps: int):
    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        print(f"Processing folder: {folder}")
        output_folder = os.path.join(output_dir, folder)
        os.makedirs(output_folder, exist_ok=True)

        frame_number = 0
        for video_file in os.listdir(folder_path):
            if not video_file.lower().endswith(VIDEO_EXTENSIONS):
                print(f"Skipping non-video file: {video_file}")
                continue

            video_path = os.path.join(folder_path, video_file)
            print(f"Processing video: {video_file}")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Unable to open video {video_path}")
                continue

            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = max(1, int(original_fps / fps))

            frame_count = 0
            video_frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    frame_number += 1
                    video_frame_count += 1
                    cv2.imwrite(os.path.join(output_folder, f"{frame_number}.jpg"), frame)

                frame_count += 1

            cap.release()
            print(f"Frames extracted from {video_file}: {video_frame_count} frames")

    print("Video to frames conversion complete.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Folder of per-class video subfolders")
    parser.add_argument("--output-dir", required=True, help="Folder to write per-class frame images into")
    parser.add_argument("--fps", type=int, default=15, help="Frames to extract per second of video")
    args = parser.parse_args()

    video_to_frames(args.input_dir, args.output_dir, args.fps)


if __name__ == "__main__":
    main()
