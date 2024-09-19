# Lumbar Spine MRI Segmentation and Disc Pathology Prediction
This project aims to extract latent features from segmented lumbar spine MRIs and utilize them to predict disc pathologies, such as disc bulging. Additionally, we provide an interpretation of the latent features (Link all'abstract)

The project is divided into two parts:

**1) Segmentation of Lumbar Spine Intervertebral Discs:** <br>
Using an open-source dataset (see [dataset link or paper]), we perform segmentation of lumbar spine intervertebral discs

**2) Latent Feature Extraction and Disc Bulging Prediction:** <br>
A 3D autoencoder is trained to extract latent features from the predicted disc masks, which are then used to predict disc bulging via a gradient boosting classifier. This part also includes interpretability of the latent features


The code for the first part of the project is available in this repository. You can find the model weights for the segmentation in the [weights folder](link to Google Drive)

## Overview of the Project

<br>

<p align="center">
  <img src="Figures/Figure 1.png" width="700" height="400">
</p>

<br>
Figure 1: Overview of the project pipeline






## Dataset
We use an open-source dataset of lumbar spine T2-weighted MRIs (see [dataset link or paper]). The dataset consists of 198 images from respective patients. A train/validation/test split of 60/20/20 was applied

## Preprocessing
The following preprocessing steps were performed:
* Resampling to a target resolution
* Padding/Cropping
* Image normalization/standardization

## Model
We employed the Swin Transformer (specifically, SwinUNETR from MONAI) for the segmentation task. While there are pre-trained models available (e.g., from TotalSegmentator), accurate segmentation of specific anatomical regions like the intervertebral disc remains challenging


<br>

<p align="center">
  <img src="Figures/Figure 2.png" width="700" height="450">
</p>

<br>





Figure 2: Architecture of the Swin Transformer model

## Results
For the segmentation task using the Swin Transformer, we achieved the following results:<br>
**Intersection over Union (IoU):** XXX (95% CI, 0.77–0.80)<br> 
**Dice Similarity Coefficient:** XXX (95% CI, 0.87–0.89)<br>
These results align with previous analyses of this dataset (see [link to study])

Incorporating the latent features extracted from the 3D autoencoder into a gradient boosting classifier improved disc bulging prediction compared to using geometric features alone. The performance was comparable to studies that combine geometric and anthropometric data (see tables below for detailed results)


<br>

<p align="center">
  <img src="Figures/Figure 3.png" width="700" height="100">
</p>

<br>





Please note that the code for the gradient boosting classifier is part of the second phase of the project and is not included in this repository
