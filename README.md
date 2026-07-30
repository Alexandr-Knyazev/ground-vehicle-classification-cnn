# Ground Vehicle Classification (Civilian vs Military)

Binary classification of civilian and military ground vehicles from aerial images using EfficientNet-B0 (PyTorch).  
Классификация гражданской и военной наземной техники по аэрофотоснимкам.

## Dataset structure

```text
vehicle_dataset/
  civilian/
    *.jpg
  military/
    *.jpg
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
cd src
python train.py
```

### GUI application

```bash
cd src
python gui_app.py
```

### Inference (optional)

```bash
cd src
python inference.py
```

## Project structure

```text
ground vehicle classification/
  README.md
  requirements.txt
  .gitignore
  src/
    train.py
    inference.py
    gui_app.py
  vehicle_dataset/
  models/
```

## Model and Windows binary

Model file and prebuilt Windows executable are available separately (not included in this repository).