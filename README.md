# BAM Network Project

## Overview
This project implements a Bidirectional Associative Memory (BAM) neural network designed for recognizing English uppercase letters represented in an 8x8 ASCII format. The implementation includes both a basic BAM network and an enhanced version, showcasing improvements in performance and flexibility. The network is trained on ASCII representations of letters and tested using provided datasets.

## Features
- **BAM.py**: Contains the core implementation of the basic BAM network.
- **enhanced_BAM.py**: Introduces an improved version of the BAM network, optimizing memory usage and increasing accuracy.
- **using_BAM.py**: A script to demonstrate the usage of the basic BAM network with training and testing.
- **using_enhanced_BAM.py**: Demonstrates the usage of the enhanced BAM network with training and testing.
- **English_Uppercase_8x8_ASCII_Complete.csv**: A dataset containing ASCII representations of English uppercase letters for training.
- **letters_data_test.csv**: A dataset used for testing the network's performance.

## Structure
```
BAM_network/
├── BAM.py
├── enhanced_BAM.py
├── using_BAM.py
├── using_enhanced_BAM.py
├── English_Uppercase_8x8_ASCII_Complete.csv
├── letters_data_test.csv
├── __init__.py
├── .git
├── .gitattributes
└── __pycache__/
```

### File Descriptions
- **BAM.py**: Implements the standard BAM algorithm using binary matrices for pattern storage and recall.
- **enhanced_BAM.py**: Adds enhancements such as:
  - Improved weight matrix handling.
  - Error tolerance for noisy inputs.
  - Adaptive learning for dynamic training.
- **using_BAM.py**: Demonstrates how to load data, train the basic BAM network, and test its recall capabilities.
- **using_enhanced_BAM.py**: Similar to `using_BAM.py` but utilizes the enhanced BAM network for improved results.
- **English_Uppercase_8x8_ASCII_Complete.csv**: Provides the full training set of uppercase letters in ASCII.
- **letters_data_test.csv**: Test data for validation.
- **__init__.py**: Marks the directory as a Python package.
- **.git**: Git repository folder for version control.
- **.gitattributes**: Configuration for Git's behavior with certain file types.

## Enhancements in `enhanced_BAM.py`
- **Dynamic Thresholding**: Allows the network to handle a wider range of input noise.
- **Efficient Weight Matrix Updates**: Reduces redundancy and improves memory usage.
- **Robust Testing**: Enhanced ability to generalize and recall patterns under noisy conditions.

## Usage
### Prerequisites
- Python 3.7 or higher.
- Required libraries: NumPy, pandas.

Install the required libraries using pip:
```bash
pip install numpy pandas
```

### Training and Testing
1. **Basic BAM Network**:
   Run the `using_BAM.py` script:
   ```bash
   python using_BAM.py
   ```
   This script loads the training data, trains the BAM network, and tests its recall capabilities.

2. **Enhanced BAM Network**:
   Run the `using_enhanced_BAM.py` script:
   ```bash
   python using_enhanced_BAM.py
   ```
   This script showcases the improved recall performance with the enhanced BAM.

### Input and Output
- **Input**: ASCII representations of English uppercase letters.
- **Output**: Predicted letters or error metrics.

## Example
### Basic BAM Network
Input: Noisy representation of `A`.
Output: Correct recall of `A`.

### Enhanced BAM Network
Input: Heavily distorted representation of `B`.
Output: Accurate recall of `B` due to error tolerance.
