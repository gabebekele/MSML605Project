import os
import tensorflow as tf
import numpy as np
from pathlib import Path

def load_image_vector(path_str, lfw_dir, target_shape=(32, 32), normalization=None):
    base_path = Path(lfw_dir)
    
    # Clean the incoming path
    relative_path = Path(path_str.replace("\\", "/"))
    
    """
    # prevent double lfw in path
    if relative_path.parts[0] == "lfw" and base_path.name == "lfw":
        relative_path = Path(*relative_path.parts[1:])
"""

    full_path = base_path / relative_path

    if not full_path.exists():
        raise FileNotFoundError(f"Image not found at: {full_path.absolute()}")

    # load images
    img_raw = tf.io.read_file(str(full_path))
    img_tensor = tf.image.decode_jpeg(img_raw, channels=3)
    img_tensor = tf.image.resize(img_tensor, target_shape)
    img_tensor = tf.cast(img_tensor, tf.float32)

    # use norm for data-centric improvement 
    if normalization == "z-score":
        mean, variance = tf.nn.moments(img_tensor, axes=[0, 1, 2])
        img_tensor = (img_tensor - mean) / tf.sqrt(variance + 1e-7)
    elif normalization == "min-max":
        img_tensor = img_tensor / 255.0

    return tf.reshape(img_tensor, [-1]).numpy()

def embed_pair(left_path, right_path, config, normalization=None):
    
    lfw_dir = config["data_dir"]
    
    # turn images to vectors
    leftvec = load_image_vector(left_path, lfw_dir, normalization=normalization)
    rightvec = load_image_vector(right_path, lfw_dir, normalization=normalization)
    
    return leftvec, rightvec
