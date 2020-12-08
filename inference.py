import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from basenet import model
import opencv as cv2


# returns : The region and the affinity score map for a given image on the trained model
# input : an image
def inference(img):
    img = img/255.0
    img = img-np.mean(img)
    img = cv2.imread(img)
    img = cv2.resize(img, (512, 512, 3))
    img = img.reshape((1, 512, 512, 3))
    return model.predict(img)

# Visualising the score_map using a plot
# input : a score_map
def visualise(score_map):
    plt.pcolormesh(score_map)
    return