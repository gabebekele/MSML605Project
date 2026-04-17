import os
import tensorflow as tf
import numpy as np
from pathlib import Path
import torch
import tensorflow_datasets as tfds
from facenet_pytorch import InceptionResnetV1

# Load model once
MODEL = InceptionResnetV1(pretrained="vggface2").eval()

IMAGE_CACHE = None


def build_image_cache(lfw_dir):
    global IMAGE_CACHE

    if IMAGE_CACHE is not None:
        return IMAGE_CACHE

    ds = tfds.load("lfw", split="train", data_dir=lfw_dir)

    image_cache = {}
    per_id_index = {}

    for ex in tfds.as_numpy(ds):
        identity = ex["label"].decode("utf-8")

        per_id_index[identity] = per_id_index.get(identity, 0) + 1
        filename = f"{identity}_{per_id_index[identity]:04d}.jpg"
        rel_path = os.path.join("lfw", identity, filename).replace("\\", "/")

        image_cache[rel_path] = ex["image"]

    IMAGE_CACHE = image_cache
    return IMAGE_CACHE


def load_image_vector(path_str, lfw_dir, target_shape=(160, 160), normalization=None):
    image_cache = build_image_cache(lfw_dir)

    relative_path = path_str.replace("\\", "/")

    if relative_path not in image_cache:
        raise FileNotFoundError(f"Image not found for key: {relative_path}")

    img_array = image_cache[relative_path]

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    img_tensor = tf.image.resize(img_tensor, target_shape)

    # use norm for data-centric improvement
    if normalization == "z-score":
        mean, variance = tf.nn.moments(img_tensor, axes=[0, 1, 2])
        img_tensor = (img_tensor - mean) / tf.sqrt(variance + 1e-7)
    elif normalization == "min-max":
        img_tensor = img_tensor / 255.0
    else:
        img_tensor = img_tensor / 255.0

    # convert pytorch tensor for embedding
    img_array = img_tensor.numpy().astype(np.float32)
    img_tensor_pt = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        embedding = MODEL(img_tensor_pt)

    return embedding.squeeze(0).numpy()


def embed_pair(left_path, right_path, config, normalization=None):
    lfw_dir = config["data_dir"]

    # turn images to vectors
    leftvec = load_image_vector(left_path, lfw_dir, normalization=normalization)
    rightvec = load_image_vector(right_path, lfw_dir, normalization=normalization)

    return leftvec, rightvec
