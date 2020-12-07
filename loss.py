import tensorflow as tf

# Usage : loss(data)
# Input : List of arrays of region_true, affinity_true, region_pred, affinity_pred, confidence map (Pixelwise)
# return : A scalar MSE Loss

def loss(data):
    region_true, affinity_true, region_pred, affinity_pred, confidence = data

    region_loss = (region_true-region_pred)**2
    region_loss *= confidence

    affinity_loss = (affinity_true-affinity_pred)**2
    affinity_loss *= confidence

    total_loss = region_loss+affinity_loss

    return tf.reduce_sum(total_loss)