# EfficientNet Classification Project

This project implements an image classification model using the EfficientNet architecture with pre-trained weights. The model is designed to classify images from a specified dataset.

## Project Structure

```
efficientnet-classification
├── src
│   ├── data
│   │   ├── dataset.py        # Handles loading and preprocessing of the dataset
│   │   └── transforms.py     # Defines data transformation functions
│   ├── models
│   │   ├── efficientnet.py   # Imports EfficientNet model architecture
│   │   └── head.py           # Defines the classification head model
│   ├── train.py              # Contains the training loop for the model
│   ├── evaluate.py           # Functions for evaluating the trained model
│   └── utils.py              # Utility functions used throughout the project
├── requirements.txt          # Lists project dependencies
├── config.yaml               # Configuration settings for the project
└── README.md                 # Documentation for the project
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd efficientnet-classification
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Configuration

Edit the `config.yaml` file to set paths to your dataset and adjust training parameters as needed.

## Usage

### Training the Model

To train the model, run the following command:
```
python src/train.py
```

### Evaluating the Model

To evaluate the trained model, use:
```
python src/evaluate.py
```

## Acknowledgments

- EfficientNet model architecture by Google AI.
- PyTorch and torchvision for deep learning framework and datasets.
