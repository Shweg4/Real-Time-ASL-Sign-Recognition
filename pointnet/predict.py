"""Evaluate the trained PointNet model on a held-out image set: accuracy, confusion matrix, and misclassified-sample dump."""
import itertools
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix

from utils import char2int, clean_folder, get_hand_points, load_config

device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def predict(model, img):
    model.eval()
    points_raw = get_hand_points(img)
    try:
        points = points_raw.copy()
        min_x, max_x = np.min(points_raw[:, 0]), np.max(points_raw[:, 0])
        min_y, max_y = np.min(points_raw[:, 1]), np.max(points_raw[:, 1])
        for i in range(len(points_raw)):
            points[i][0] = (points[i][0] - min_x) / (max_x - min_x)
            points[i][1] = (points[i][1] - min_y) / (max_y - min_y)
    except (TypeError, ValueError):
        return None, None

    points_t = torch.tensor([points]).float().to(device)
    label = model(points_t)
    label = label.detach().cpu().numpy()
    label = np.argmax(label)
    label = list(char2int.keys())[list(char2int.values()).index(label)]

    return label, points_raw


def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues, file=""):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True Sign")
    plt.xlabel("Predicted Sign")
    plt.savefig(file)
    plt.show()


def display_histogram(sign_dict):
    sign_dict = dict(sorted(sign_dict.items()))
    plt.bar(list(sign_dict.keys()), sign_dict.values(), color="steelblue", edgecolor="black", linewidth=1.2)
    plt.title("Distribution of misclassified signs")
    plt.xlabel("Sign")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)
    plt.show()


def predict_images(images_per_class=100):
    config = load_config("config.yaml")
    model_name = config["model"]["name"]
    path = config["dataset"]["test_dataset"]
    results_path = config["paths"]["results_path"]
    missclassified_path = config["paths"]["missclassified_path"]
    model_path = config["model"]["model_path"]

    model = torch.load(os.path.join(model_path, model_name), map_location=torch.device("cpu"))

    clean_folder(missclassified_path)
    os.makedirs(missclassified_path, exist_ok=True)

    actuals, predicteds = [], []
    signs = list(char2int.keys())
    wrongs = {}

    for root, _dirs, files in os.walk(path):
        gt = root.split(os.sep)[-1].lower()
        if gt not in signs:
            continue

        print(f"Current sign: {gt}")
        for count, file in enumerate(files):
            if count >= images_per_class:
                break

            img = cv2.imread(os.path.join(root, file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img.flags.writeable = False
            predicted_label, _points = predict(model, img)

            if predicted_label is None:
                continue

            predicteds.append(predicted_label)
            actuals.append(gt)
            if predicted_label != gt:
                wrongs[predicted_label] = wrongs.get(predicted_label, 0) + 1
                os.makedirs(os.path.join(missclassified_path, predicted_label), exist_ok=True)
                cv2.imwrite(
                    os.path.join(missclassified_path, predicted_label, f"{gt}_{file}"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                )

    acc = accuracy_score(actuals, predicteds)
    print(f"Accuracy: {acc:.4f}")

    cm = confusion_matrix(actuals, predicteds, labels=signs)
    plot_confusion_matrix(cm, signs, normalize=False, file=os.path.join(results_path, "confusion_matrix.png"), cmap=plt.cm.Purples)

    print(f"{sum(wrongs.values())} misclassified images")
    display_histogram(wrongs)


if __name__ == "__main__":
    predict_images()
