# FER-ResNet-50: Facial Expression Recognition with ResNet-50

A deep learning implementation for recognizing facial expressions using ResNet-50 architecture.

## Overview

This project implements a Facial Expression Recognition (FER) system using a pre-trained ResNet-50 model. The model is fine-tuned to classify facial expressions into different emotion categories.

## Features

- ResNet-50 backbone architecture
- Transfer learning approach
- Support for multiple emotion classes
- Data preprocessing pipeline
- Model training and evaluation utilities
- Real-time facial expression detection (optional)

## Project Structure

```
FER-ResNet-50/
├── data/                    # Dataset directory
│   ├── raw/                # Raw image data
│   └── processed/          # Processed images
├── models/                 # Model checkpoints
├── notebooks/              # Jupyter notebooks for experimentation
├── src/                    # Source code
│   ├── __init__.py
│   ├── model.py           # Model architecture
│   ├── train.py           # Training script
│   ├── evaluate.py        # Evaluation script
│   ├── dataset.py         # Data loading utilities
│   └── utils.py           # Helper functions
├── requirements.txt        # Project dependencies
├── .gitignore             # Git ignore file
└── README.md              # Project documentation
```

## Requirements

- Python 3.8+
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- numpy
- pandas
- opencv-python
- matplotlib
- scikit-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/RajBhadani/FER-ResNet-50.git
cd FER-ResNet-50
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python src/train.py --epochs 50 --batch-size 32 --lr 0.001
```

### Evaluation

```bash
python src/evaluate.py --model-path models/best_model.pth --test-data data/test/
```

### Inference

```python
from src.model import FERResNet50
from src.utils import preprocess_image

model = FERResNet50(num_classes=7)
model.load_state_dict(torch.load('models/best_model.pth'))

image = preprocess_image('face_image.jpg')
prediction = model(image)
```

## Datasets

This project can work with various facial expression datasets:
- FER2013
- CK+ (Extended Cohn-Kanade)
- AffectNet
- RAF-DB

## Results

*(Results to be updated after training)*

## Contributing

Contributions are welcome! Please feel free to submit issues and enhancement requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- RajBhadani

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR.
- [PyTorch ResNet Documentation](https://pytorch.org/vision/stable/models.html#resnet)

---

For questions or issues, please open an issue on the repository.
