# Bayesian Surprise Model Integration

This project integrates a Bayesian surprise model into the original time without clocks model. Link to original model [here](https://www.nature.com/articles/s41467-018-08194-7#Abs1). The goal is to enhance the prediction of video duration by focusing on the most salient regions of each video frame, simulating human attention.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Testing](#testing)
- [Code Overview](#code-overview)
  - [Saliency Model](#saliency-model)
  - [Feature Extraction](#feature-extraction)
  - [Main Script](#main-script)
- [Results](#results)
- [License](#license)

## Introduction

This project uses a Bayesian surprise model to predict areas in video frames that humans are more likely to look at(Salient regions). By focusing on these regions, the model can more accurately estimate the perceived duration of the video, as it aligns with how human attention typically processes visual information. The model is based on the Bayesian surprise mechanism as outline in the research papers linked [here](https://www.sciencedirect.com/science/article/abs/pii/S0893608009003256) and [here](https://www.researchgate.net/publication/23299422_Bayesian_Surprise_Attracts_Human_Attention), which helps to predict regions in the video frames that are probably more likely to be surprising or attention-grabbing.

## Features

- **Bayesian Saliency Model**: Predicts the most salient regions in video frames.
- **Time Estimation Model**: Uses AlexNet to extract features from the spotlighted regions and predicts the perceived duration of the video.
- **Integration**: The saliency model is seamlessly integrated with the time estimation model to improve prediction accuracy.
- **Visualization**: The code includes functionality to plot and compare the predicted versus actual video durations.

## Requirements

- Python 3.6 or higher
- PyTorch 1.8 or higher
- OpenCV
- NumPy
- Matplotlib
- torchvision

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ucwealth/Bayes.git
   cd Bayes
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure that you have a CUDA-enabled GPU if you want to run the model on a GPU.

## Usage

### Training

1. Prepare a dataset of videos in the `../videos` directory (relative to the project directory).
2. Run the training process using the following command:

   ```bash
   python integrated.py
   ```

   The script will train the model on 70% of the videos and test it on the remaining 30%.

### Testing

1. The script will automatically test the model after training, predicting the duration of the test videos and plotting the results.
2. The predicted versus actual durations will be displayed in the console and visualized in a plot.

## Code Overview

### Saliency Model

- **SaliencyPredictor**: A neural network model that uses a backbone (e.g., VGG16) to extract features from consecutive video frames. It calculates Bayesian surprise to generate a saliency map, highlighting the most attention-grabbing regions of the frame.

### Feature Extraction

- **apply_saliency_spotlight**: This function applies the saliency map to crop the video frame around the most salient region, ensuring the model focuses on areas most likely to attract human attention.
- **extract_features_with_saliency**: This function processes each video frame, applying the saliency spotlight filter, and extracts features using the AlexNet model.

### Main Script

- The main script (`main.py`) orchestrates the entire process:
  - Initializes models, including the saliency predictor and time estimation model.
  - Processes videos to extract features using the saliency model.
  - Trains the time estimation model on the processed features.
  - Tests the model and plots the predicted versus actual durations.

## Results

The model is expected to produce more accurate time estimates by focusing on the most salient regions of each video frame. The `plot_durations` function provides a visual comparison of predicted and actual video durations, which helps assess the model's performance.


