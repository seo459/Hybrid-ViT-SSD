# Hybrid ViT-SSD

## Table of Contents
1. [Introduction]
2. [Background]
   - [Vision Transformer (ViT)]
   - [Attention Map]
   - [Comparison of Inference Speed]
   - [Single Shot multibox Detection (SSD)]
   - [NMS (Non-Maximum Suppression) Algorithm]
3. [Dataset and Method]
   - [Dataset]
   - [Method]
4. [Results]
   - [Environment]
   - [SSD]
   - [ViT]
5. [Discussion]
6. [References]

## Introduction
This project focuses on upgrading the Vision Transformer (ViT) model, for image classification. To enhance the performance of the ViT model, we implement an additional preprocessing step using the SSD model before the Input Embedding process.

## Background

### Vision Transformer (ViT)
ViT is composed of an encoder and decoder structure, using input that has been through the embedding process as input for Multihead Attention (MSA). This approach observes the overall context rather than focusing on a specific point, especially when dealing with images requiring high accuracy.

### Attention Map
Attention maps visually identify the areas being attended to in the image. They help verify whether the model is focusing on the correct regions.

### Comparison of Inference Speed
The inference speed of ViT is highly dependent on the input size. Smaller images improve the inference speed but there is a trade-off with the time taken for preprocessing.

### Single Shot multibox Detection (SSD)
SSD generates anchor box centers at reasonable locations, reducing unnecessary computations. This approach is used to crop image areas that will be input to ViT.

### NMS (Non-Maximum Suppression) Algorithm
NMS eliminates bounding boxes with low confidence scores and those with high Intersection over Union (IoU) values with others, helping to accurately identify objects.

## Dataset and Method

### Dataset
We used Roboflow to classify and set the ground truth for 671 original images, which were then augmented to create a dataset of 1548 training, 80 validation, and 80 testing images.

### Method
1. Use SSD to crop the image area that will be input to ViT.
2. Insert the bounding box produced by SSD as an input to ViT.

This process aims to enhance the prediction accuracy of the ViT model by focusing on images containing a single object, similar to noise reduction.

## Results

### Environment
- **Platform**: Google Colab Pro
- **System RAM**: 50GB
- **GPU**: NVIDIA A100, V100, T4

### SSD
- Validation Loss and Accuracy graphs

### ViT
- Validation Loss, Validation Accuracy, Training Loss, and Learning Rate graphs

## Discussion
The main advantage of using ViT with SSD's bounding boxes is the potential increase in inference speed due to reduced input size. However, using SSD slows down the process. The primary goal is accurate analysis of specific objects, not real-time inference.

Challenges faced:
1. Inconsistent ground truth criteria leading to a less effective dataset.
2. Insufficient amount of training data for the ViT model, even after augmentation.
3. Inability to use pre-trained models, which are recommended for better performance.

The project highlighted the need for substantial computing resources, with long training times (up to 30 hours) making it difficult to conduct experiments efficiently.

## References
1. [ViT baseline](https://github.com/YoojLee/vit)
2. [SSD baseline](https://d2l.ai/chapter_computer-vision/ssd.html)

