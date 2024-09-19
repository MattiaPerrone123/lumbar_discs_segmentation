# Lumbar Spine MRI Segmentation and Disc Pathology Prediction
This project aims to extract latent features from segmented lumbar spine MRIs and utilize them to predict disc pathologies, such as disc bulging. Additionally, we provide an interpretation of the latent features. The current study was currently accepted at PSRS-ORS 2024 as a [podium presentation](./Papers/PSRS_ORS_2024.pdf) and we are currently finishing to write the paper

The project is divided into two parts:

**1) Segmentation of Lumbar Spine Intervertebral Discs:** <br>
Using an [open-source dataset](https://doi.org/10.5281/zenodo.8009680), we performed segmentation of lumbar spine intervertebral discs

**2) Latent Feature Extraction and Disc Bulging Prediction:** <br>
A 3D autoencoder is trained to extract latent features from the predicted disc masks, which are then used to predict disc bulging via a gradient boosting classifier. This part also includes interpretability of the latent features


The code for the first part of the project is available in this repository. You can find the model weights for the segmentation [here] (https://drive.google.com/file/d/1D44Om1X1bx4eWHOTtpJWKG4gp9OGl-Fo/view?usp=sharing)

## Overview of the Project

<br>

<p align="center">
  <img src="Figures/Figure 1.png" width="700" height="430">
</p>

<br>





## Dataset
We used an [open-source dataset](https://doi.org/10.5281/zenodo.8009680) of lumbar spine T2-weighted MRIs. The dataset consists of 198 images from respective patients. A train/validation/test split of 60/20/20 was applied

## Preprocessing
The following preprocessing steps were performed:
* Resampling to a target resolution
* Padding/Cropping
* Image normalization/standardization

## Model
We employed the Swin Transformer (specifically, SwinUNETR from MONAI) for the segmentation task. While there are pre-trained models available (e.g., from TotalSegmentator), accurate segmentation of specific anatomical regions like the intervertebral disc remains challenging


<br>

<p align="center">
  <img src="Figures/Figure 2.png" width="700" height="400">
</p>

<br>





Figure 2: Architecture of the Swin Transformer model

## Results
For the segmentation task using the Swin Transformer, we achieved the following results:<br>
**Intersection over Union (IoU):** 0.79 (95% CI, 0.77–0.80)<br> 
**Dice Similarity Coefficient:** 0.88 (95% CI, 0.87–0.89)<br>
These results align with [previous analyses](./Papers/Van_der_graaf_2023) of this dataset 

Incorporating the latent features extracted from the 3D autoencoder into a gradient boosting classifier improved disc bulging prediction compared to using geometric features alone. The performance was comparable to studies that combine geometric and anthropometric data (see tables below for detailed results)


<br>

<p align="center">
  <img src="Figures/Figure 3.png" width="900" height="90">
</p>

<br>





Please note that the code for the gradient boosting classifier is part of the second phase of the project and is not included in this repository
