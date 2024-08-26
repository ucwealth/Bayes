import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
from torchvision import transforms
import cv2
import sys, os 


class GazeDataset(Dataset):
    """
    Custom Dataset for loading video frames and corresponding gaze maps.

    Args:
        frames (list of str): List of paths to video frames.
        gaze_maps (list of str): List of paths to corresponding gaze maps.
        transform (callable, optional): A function/transform to apply to the frames.
    """
    def __init__(self, frames, gaze_maps, transform=None):
        self.frames = frames
        self.gaze_maps = gaze_maps
        self.transform = transform

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        
        frame = np.load(self.frames[idx])
        gaze_map = np.load(self.gaze_maps[idx])
        
        # Apply transform if available
        if self.transform:
            frame = self.transform(frame)
        else:
            frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)

        gaze_map = torch.tensor(gaze_map, dtype=torch.float32)
        return frame, gaze_map

# Define data augmentation and normalization
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class FeatureDetectionFrontEnd(nn.Module):
    """
    Front-end neural network component for extracting visual features.
    """
    def __init__(self):
        super(FeatureDetectionFrontEnd, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        return x


class SurpriseComputationBackEnd(nn.Module):
    """
    Back-end neural network component for computing surprise based on features.
    """
    def __init__(self):
        super(SurpriseComputationBackEnd, self).__init__()
        self.fc1 = nn.Linear(128 * 32 * 32, 256)  # Adjust dimensions based on input size
        self.fc2 = nn.Linear(256, 1)  # Output a single surprise score

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Sigmoid for output in range [0, 1]
        return x


class BayesianSurpriseModel(nn.Module):
    """
    Complete neural network model combining feature extraction and surprise computation.
    """
    def __init__(self):
        super(BayesianSurpriseModel, self).__init__()
        self.feature_detector = FeatureDetectionFrontEnd()
        self.surprise_computer = SurpriseComputationBackEnd()

    def forward(self, x):
        features = self.feature_detector(x)
        surprise = self.surprise_computer(features)
        return surprise


def train_model(model, dataloader, criterion, optimizer, scheduler=None, num_epochs=10):
    """
    Train the neural network model.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): The DataLoader for training data.
        criterion (nn.Module): The loss function.
        optimizer (Optimizer): The optimizer for training.
        scheduler (LRScheduler, optional): Learning rate scheduler.
        num_epochs (int): Number of epochs to train.
    """
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in dataloader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        if scheduler:
            scheduler.step()  # Adjust learning rate

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')


def evaluate_model(model, dataloader, criterion):
    """
    Evaluate the neural network model.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): The DataLoader for validation or test data.
        criterion (nn.Module): The loss function.

    Returns:
        float: The loss over the dataset.
        float: AUC score of the model predictions.
    """
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    val_loss = running_loss / len(dataloader.dataset)
    auc = roc_auc_score(np.concatenate(all_targets), np.concatenate(all_predictions))
    return val_loss, auc


def save_checkpoint(model, optimizer, epoch, filepath='checkpoint.pth'):
    """
    Save model and optimizer state to a checkpoint file.

    Args:
        model (nn.Module): The model to save.
        optimizer (Optimizer): The optimizer state to save.
        epoch (int): The current epoch number.
        filepath (str): File path to save the checkpoint.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)


def load_checkpoint(filepath, model, optimizer):
    """
    Load model and optimizer state from a checkpoint file.

    Args:
        filepath (str): File path to load the checkpoint from.
        model (nn.Module): The model to load the state into.
        optimizer (Optimizer): The optimizer to load the state into.

    Returns:
        nn.Module: Model with loaded state.
        Optimizer: Optimizer with loaded state.
        int: The epoch number at which the checkpoint was saved.
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


def convert_images_and_maps_to_npy(image_folder, map_folder, output_image_folder, output_map_folder):
    """
    Convert images and saliency maps in a folder to .npy files.

    Args:
        image_folder (str): Path to the folder containing input images.
        map_folder (str): Path to the folder containing saliency maps.
        output_image_folder (str): Path to the folder where .npy files for images will be saved.
        output_map_folder (str): Path to the folder where .npy files for maps will be saved.
    """
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_map_folder, exist_ok=True)

    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg"):
            # Convert image to .npy
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            npy_image_filename = os.path.join(output_image_folder, filename.replace('.jpg', '.npy'))
            np.save(npy_image_filename, image_rgb)

    for filename in os.listdir(map_folder):
        if filename.endswith(".png"):
            # Convert map to .npy
            map_path = os.path.join(map_folder, filename)
            saliency_map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
            npy_map_filename = os.path.join(output_map_folder, filename.replace('.png', '.npy'))
            np.save(npy_map_filename, saliency_map)


if __name__ == "__main__":
    # Initialize dataset with transform
    frames_dir = 'SALICON/images/train'
    map_folder = 'SALICON/maps/train'
    output_image_folder = './SALICON_npy/train/images_npy'
    output_map_folder = './SALICON_npy/train/maps_npy'
    convert_images_and_maps_to_npy(frames_dir, map_folder, output_image_folder, output_map_folder)

    dataset = GazeDataset(output_image_folder, output_map_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize model, loss function, optimizer, and scheduler
    model = BayesianSurpriseModel()
    criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Load data
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Train the model
    train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs=10)

    # Evaluate the model
    val_loss, val_auc = evaluate_model(model, dataloader, criterion)
    print(f'Validation Loss: {val_loss:.4f}, AUC: {val_auc:.4f}')

    # Save model checkpoint
    save_checkpoint(model, optimizer, epoch=10, filepath='checkpoint.pth')

    # Load model checkpoint
    model, optimizer, start_epoch = load_checkpoint('checkpoint.pth', model, optimizer)

