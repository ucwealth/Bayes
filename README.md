# Bayesian Surprise Model Integration

This project integrates a Bayesian surprise model into the original time without clocks model. Link to original model [here](https://www.nature.com/articles/s41467-018-08194-7#Abs1). The goal is to enhance the model's ability to simulate human predictions of video duration by focusing on the most salient regions of each video frame.

Here is a comprehensive README for your code that covers all the key aspects a user would need to understand and run your model effectively:

---

# Bayesian Surprise Model for Gaze Prediction

This project implements a Bayesian surprise model to predict gaze patterns based on video frames and corresponding gaze maps. The model is designed using PyTorch and leverages convolutional neural networks to extract visual features and compute surprise scores for each frame, predicting regions of interest where human gaze is most likely to be focused.

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Dataset Preparation](#dataset-preparation)
4. [Code Structure](#code-structure)
5. [Usage](#usage)
6. [Training the Model](#training-the-model)
7. [Evaluating the Model](#evaluating-the-model)
8. [Saving and Loading Checkpoints](#saving-and-loading-checkpoints)
9. [Results](#results)
10. [License](#license)

## Overview

This repository contains code for a Bayesian surprise-based model to predict gaze patterns. The model consists of two main components:

1. **Feature Detection Front-End**: A convolutional neural network (CNN) for extracting visual features from input video frames.
2. **Surprise Computation Back-End**: A neural network for computing surprise scores based on the extracted features.

The model is trained using the SALICON dataset, which provides images and corresponding saliency maps representing human gaze patterns.

## Requirements

To run this code, you need the following libraries installed:

- Python 3.7+
- PyTorch
- torchvision
- numpy
- scikit-learn
- OpenCV
- matplotlib (optional, for visualization)

You can install these dependencies using pip:

```bash
pip install torch torchvision numpy scikit-learn opencv-python matplotlib
```

## Dataset Preparation

The model uses the SALICON dataset for training and evaluation. You need to convert the images and saliency maps to `.npy` format for efficient loading during training. 

### Step-by-Step Instructions

1. **Download the SALICON Dataset**:
   - Download the dataset from [SALICON](http://salicon.net/).
   - Place the images in a folder named `SALICON/images/train` and saliency maps in `SALICON/maps/train`.

2. **Convert Images and Maps to .npy Format**:
   - Use the `convert_images_and_maps_to_npy` function provided in the code to convert images and maps to `.npy` files:

```python
convert_images_and_maps_to_npy(image_folder='SALICON/images/train', 
                               map_folder='SALICON/maps/train',
                               output_image_folder='./SALICON_npy/train/images_npy', 
                               output_map_folder='./SALICON_npy/train/maps_npy')
```

## Code Structure

- **GazeDataset**: Custom PyTorch Dataset class for loading video frames and gaze maps.
- **FeatureDetectionFrontEnd**: CNN model for feature extraction.
- **SurpriseComputationBackEnd**: Neural network model for computing surprise scores.
- **BayesianSurpriseModel**: Combined model integrating feature detection and surprise computation.
- **train_model**: Function to train the model.
- **evaluate_model**: Function to evaluate the model's performance.
- **save_checkpoint**: Function to save model checkpoints.
- **load_checkpoint**: Function to load model checkpoints.
- **convert_images_and_maps_to_npy**: Utility function to convert images and maps to `.npy` format.

## Usage

1. **Prepare the Dataset**: Convert your images and saliency maps to `.npy` files using the provided function.
2. **Initialize the Dataset and Dataloader**:

```python
dataset = GazeDataset(frames=output_image_folder, gaze_maps=output_map_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
```

3. **Initialize the Model, Loss Function, and Optimizer**:

```python
model = BayesianSurpriseModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
```

## Training the Model

To train the model, run the following code:

```python
train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs=10)
```

The training process will print the loss at each epoch.

## Evaluating the Model

After training, you can evaluate the model using:

```python
val_loss, val_auc = evaluate_model(model, dataloader, criterion)
print(f'Validation Loss: {val_loss:.4f}, AUC: {val_auc:.4f}')
```

## Saving and Loading Checkpoints

To save a model checkpoint:

```python
save_checkpoint(model, optimizer, epoch=10, filepath='checkpoint.pth')
```

To load a model checkpoint:

```python
model, optimizer, start_epoch = load_checkpoint('checkpoint.pth', model, optimizer)
```

## Results

The model's performance is evaluated using Mean Squared Error (MSE) loss and the Area Under the Curve (AUC) of the Receiver Operating Characteristic (ROC) curve. The results indicate the model's ability to predict regions of interest in video frames based on human gaze patterns.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---
