import tensorflow as tf
from gaussian import GaussianGenerator
from net import model
from affinity_util import *
from pseudo_util import  *
from loss import loss
import numpy as np

def train(synth_x, char_boxes, real_x, real_y):
    optimizer = tf.keras.optimizers.Adam()
    g = GaussianGenerator
    h, w =  synth_x[0].shape[:2]
    region_score = np.array([g.gen((h//2, w//2), x) for x in char_boxes])
    affinity_boxes = [[cal_affinity_boxes(y) for y in char_box] for char_box in char_boxes]
    affinity_score = [g.gen((h//2, w//2), x) for x in affinity_boxes]
    affinity_score = np.array(affinity_score)
    model.compile(loss="mse", optimizer=optimizer, metrics="acc")
    epochs = 100
    for epoch in range(epochs):
        model.fit(synth_x, [region_score, affinity_score], epochs = 1)
        with tf.GradientTape() as tape:
            pseudo_region = model(real_x)[0]
            pseudo_char_conf = [gen(img, real_y[0], real_y[1]) for img in pseudo_region] #pass word annotations also in this
                                                                                       #returns char boxes and confidence map
            pseudo_region_score = np.array([g.gen((h//2, w//2), x) for x in pseudo_char_conf[:,0]])
            pseudo_affinity_boxes = [[cal_affinity_boxes(y) for y in pseudo_char_conf_word[:,0]] for pseudo_char_conf_word in pseudo_char_conf]
            pseudo_affinity_score = [g.gen((h//2, w//2, x) for x in pseudo_affinity_boxes)]
            pseudo_affinity_score = np.array(pseudo_affinity_boxes)
            pred = model(real_x)
            total_loss = loss(pseudo_region_score, pseudo_affinity_score, pred[0], pred[1], pseudo_char_conf[:,1])
            final_loss = tf.sum(total_loss)
        
        gradients = tape.gradient(final_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))