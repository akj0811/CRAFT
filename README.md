# CRAFT

CRAFT is a framework proposed in 2019 by the Clova AI Research group for scene text detection task. CRAFT stands for Character Region Awareness for Text Detection. It is one of the state of the art models for text detection. This repository contains our implementation of the CRAFT framework in tensorflow. 

**Link to the Paper - [CRAFT](https://arxiv.org/pdf/1904.01941.pdf)**

### References

* [Keras Implementation of CRAFT](https://github.com/RubanSeven/CRAFT_keras)
* [Clova AI Group Repository](https://github.com/clovaai/CRAFT-pytorch)

## Overview

An efficient implementation of the CRAFT Text Detector. It detects text by exploring the character region and the affinity between charcters in a text instance. It has been implemented in Tensorflow-Keras.

## Getting Started

The follwing are the details of the various code files present in the repository -

* **affinity_util.ipynb** - It implements the algorithm for generation of affinity boxes between the characters given the character level bounding boxes in a text instance. 
* **gaussian.ipynb** - It generates an isotropic Gaussian Heatmap which is theb persepctively transformed to generate heatmaps for region and affinity scores.
* **loss.py** - It implements the standard Mean Squared Error loss function adapted to the use case in CRAFT Text Detection.
* **net.py** - It implements the VGG-16 support network for the extraction of features from the images. It is adapted by adding upconv blocks to meet the requirement of the framework.
* **pseudo_util.ipyb** - It implements the Pseudo ground truth generation pipeline which is needed since the framework is weakly supervised. It returns the lables for the images based on an interim model.
* **train.py** - It implements the training pipline that essentially clubs the utilities implemented. It merges the weak supervision as well as the training on synthetic dataset.
* **inference.py** - It implements the inference pipeline for generating heatmaps for any given image containing text instances.
* **pre_processing.ipynb** - Since, synthtext dataset doesn't directly provide us the character boxes for a particular word in the image. We need to pre-process the dataset before feeding the data to our model.
