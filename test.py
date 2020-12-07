import tensorflow as tf
import numpy as np
from net import model
import matplotlib.pyplot as plt

def infer(img):
    model.predict(img)
    