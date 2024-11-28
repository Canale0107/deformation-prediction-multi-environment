import os
from itertools import combinations
from pathlib import Path
from datetime import datetime
import pickle
import json
from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy, BinaryFocalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

from models.handnet_based_model import handnet_based_model
from util.training import init_device
from util.training.dataloader import split_data_for_multiple_location, concat_and_shuffle
from util.training.metrics import IntersectionOverUnion, MeanPixelAccuracy

PROJECT_DIRPATH = Path('/tf/workspace/deformation-prediction-multi-environment')
NAS_DIRPATH = Path('/tf/nas/')

print("TensorFlow version:", tf.__version__)