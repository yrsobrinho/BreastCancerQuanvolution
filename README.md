# Breast Cancer Quanvolution

This repository presents a hybrid classical-quantum machine learning framework for breast cancer detection using medical imaging datasets. By combining classical and quantum techniques, the project explores new frontiers in medical image classification, offering a comparative evaluation of their effectiveness.

## Overview

This project investigates the use of quantum computing—through variational quantum circuits and quantum convolutional neural networks (QCNNs)—in the context of breast cancer detection. It provides implementations for both classical and quantum pipelines and evaluates their performance on standard datasets.

## Project Structure

```
├── checkpoints/          # Saved model checkpoints
├── classical/            # Classical ML implementations
│   ├── BCDR/             # Classical models for BCDR dataset
│   └── BreastMNIST/      # Classical models for BreastMNIST dataset
├── quantum/              # Quantum ML implementations
│   ├── BCDR/             # Quantum models for BCDR dataset
│   └── BreastMNIST/      # Quantum models for BreastMNIST dataset
├── data/                 # Dataset storage and preprocessing
│   ├── BCDR/             # BCDR dataset and utilities
│   └── utils/            # General data preprocessing tools
├── graphic/              # Visualization and analysis scripts
│   ├── combined_roc.ipynb # Notebook to plot combined ROC curves
│   ├── classical/        # ROC curve data for classical models
│   └── quantum/          # ROC curve data for quantum models
├── models/               # Trained models and architecture definitions
│   ├── classical/        # Classical model architectures
│   └── quantum/          # Quantum model architectures
├── utils/                # General utility scripts
│   ├── calculator.py     # General-purpose computation tools
│   └── normalizations.txt # Normalization parameters used
├── LICENSE               # License information
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Features

- **Classical Models**: Built with PyTorch for standard deep learning approaches.
- **Quantum Models**: Developed using PennyLane for quantum-enhanced learning.
- **Dataset Support**: Includes tools for both the BCDR dataset (BreastMNIST tools are on code).
- **Performance Visualization**: Scripts and notebooks to generate and compare ROC curves.
- **Hybrid Analysis**: Provides side-by-side performance evaluations between classical and quantum models.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/BreastCancerQuanvolution.git
cd BreastCancerQuanvolution
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Datasets

- **BCDR**: A real-world dataset of breast cancer images.
- **BreastMNIST**: A benchmark medical imaging dataset from MedMNIST.

Dataset-specific download instructions and preprocessing steps can be found in the respective `data/` subdirectories.

## Evaluation

- Run models from the `classical/` or `quantum/` folders for your dataset of interest.
- Evaluate using the ROC curve tools in the `graphic/` folder.
- Compare models using saved checkpoints and logs.

## Technologies Used

- **Quantum**: PennyLane
- **Classical**: PyTorch

## License

See the `LICENSE` file for details.

---
