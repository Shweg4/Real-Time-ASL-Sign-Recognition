"""Part 3: Run the trained model on a webcam feed, video file, or single photo."""
import argparse
import pickle

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model


def preprocess_landmarks(landmarks):
    flattened = np.array(landmarks).flatten()
    return flattened[np.newaxis, ..., np.newaxis]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to the trained .keras model")
    parser.add_argument("--label-encoder", required=True, help="Path to the pickled LabelEncoder")
    parser.add_argument("--mode", choices=["live", "video", "photo"], default="live")
    parser.add_argument("--file", help="Path to the video or photo file (required for --mode video/photo)")
    args = parser.parse_args()

    if args.mode in ("video", "photo") and not args.file:
        parser.error("--file is required when --mode is 'video' or 'photo'")

    model = load_model(args.model)
    with open(args.label_encoder, "rb") as f:
        label_encoder = pickle.load(f)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = None
    if args.mode == "live":
        cap = cv2.VideoCapture(0)
    elif args.mode == "video":
        cap = cv2.VideoCapture(args.file)

    while (args.mode in ("live", "video") and cap.isOpened()) or args.mode == "photo":
        if args.mode in ("live", "video"):
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
        else:
            frame = cv2.imread(args.file)
            if frame is None:
                print("Failed to load image.")
                break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                input_data = preprocess_landmarks(landmarks)

                prediction = model.predict(input_data, verbose=0)
                predicted_index = np.argmax(prediction)
                predicted_class = label_encoder.inverse_transform([predicted_index])[0]
                confidence = prediction[0][predicted_index]

                h, w, _ = frame.shape
                xs = [lm.x for lm in hand_landmarks.landmark]
                ys = [lm.y for lm in hand_landmarks.landmark]
                x_min, x_max = int(min(xs) * w), int(max(xs) * w)
                y_min, y_max = int(min(ys) * h), int(max(ys) * h)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"Class: {predicted_class} ({confidence:.2f})", (x_min, y_min - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3,
                )
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Sign Language Detection", frame)

        if args.mode in ("live", "video"):
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            cv2.waitKey(0)
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
