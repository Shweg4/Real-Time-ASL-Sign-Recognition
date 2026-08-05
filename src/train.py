"""Part 2: Train the CNN+Transformer hybrid model on extracted hand landmarks."""
import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import tensorflow as tf

from model import build_cnn_transformer


def load_data(pickle_path: str, test_size: float):
    df = pd.read_pickle(pickle_path)
    X = np.array([np.array(landmark).flatten() for landmark in df["landmarks"]])
    y = df["class"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    return X_train, X_test, y_train, y_test, label_encoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to the landmarks pickle (.pkl.gz) produced by extract_landmarks.py")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--model-out", default="asl_model_transformer.keras")
    parser.add_argument("--label-encoder-out", default="label_encoder.pkl")
    args = parser.parse_args()

    print("Loading data...")
    X_train, X_test, y_train, y_test, label_encoder = load_data(args.data, args.test_size)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    with open(args.label_encoder_out, "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"Label encoder saved to {args.label_encoder_out}")

    print("Building CNN + Transformer model")
    model = build_cnn_transformer(input_shape=(63, 1), num_classes=len(np.unique(y_train)))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]

    print("Training the model")
    model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test Accuracy: {test_acc:.4f}")

    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    print(classification_report(y_test, predicted_classes, target_names=label_encoder.classes_))

    model.save(args.model_out)
    print(f"Model saved to {args.model_out}")


if __name__ == "__main__":
    main()
