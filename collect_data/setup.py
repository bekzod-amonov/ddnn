# utils.py
import os
import random
import numpy as np
import tensorflow as tf

def setup_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def get_device() -> str:
    gpus = tf.config.list_physical_devices("GPU")
    return "/GPU:0" if gpus else "/CPU:0"
