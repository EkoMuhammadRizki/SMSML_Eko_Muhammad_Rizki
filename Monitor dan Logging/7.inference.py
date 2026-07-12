"""
7.Inference.py
----------------
Sends a single inference request to the locally-served MLflow model
(`mlflow models serve`) and prints the prediction. Also used to sanity-check
that the serving endpoint is reachable before wiring up Prometheus.

Prereq: serve the model first, e.g.:
    mlflow models serve -m runs:/<RUN_ID>/model -p 5001 --env-manager=local

Usage:
    python 7.Inference.py --image path/to/rice_image.jpg --url http://127.0.0.1:5001/invocations
"""

import argparse
import json
import numpy as np
import requests
from PIL import Image

CLASS_NAMES = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]


def preprocess_image(image_path, img_size=128):
    img = Image.open(image_path).convert("RGB").resize((img_size, img_size))
    arr = np.array(img) / 255.0
    return arr.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to a rice image")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:5001/invocations")
    parser.add_argument("--img_size", type=int, default=128)
    args = parser.parse_args()

    instance = preprocess_image(args.image, args.img_size)

    payload = {"inputs": [instance]}
    headers = {"Content-Type": "application/json"}

    response = requests.post(args.url, headers=headers, data=json.dumps(payload))
    response.raise_for_status()

    result = response.json()
    predictions = result.get("predictions", result)
    probs = np.array(predictions[0])
    pred_idx = int(np.argmax(probs))

    print(f"Predicted class : {CLASS_NAMES[pred_idx]}")
    print(f"Confidence      : {probs[pred_idx]:.4f}")
    print(f"Full probabilities: {dict(zip(CLASS_NAMES, [round(float(p), 4) for p in probs]))}")


if __name__ == "__main__":
    main()
