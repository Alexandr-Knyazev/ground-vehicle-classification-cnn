import os
import shutil
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 224
DEFAULT_THRESHOLD = 0.5

# Путь к модели, корень проекта = родительская папка относительно src/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_efficientnet_b0_aug2_newdataset.pth")


def load_model(model_path: str):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()
    return model


def build_transform():
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def predict_image(model, image_path: str, threshold: float = DEFAULT_THRESHOLD):
    transform = build_transform()
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_t).squeeze(1)
        prob = torch.sigmoid(output).item()
        pred = 1 if prob >= threshold else 0

    return prob, pred  # 1 = military, 0 = civilian


def classify_unlabeled_folder(
    model,
    folder_path: str,
    threshold: float = DEFAULT_THRESHOLD
):
    transform = build_transform()
    exts = [".jpg", ".jpeg", ".png", ".bmp"]

    file_list = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.splitext(f.lower())[1] in exts
    ]

    civilian_dir = os.path.join(folder_path, "civilian_pred")
    military_dir = os.path.join(folder_path, "military_pred")
    os.makedirs(civilian_dir, exist_ok=True)
    os.makedirs(military_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for path in file_list:
            img = Image.open(path).convert("RGB")
            img_t = transform(img).unsqueeze(0).to(DEVICE)
            output = model(img_t).squeeze(1)
            prob = torch.sigmoid(output).item()
            pred = 1 if prob >= threshold else 0

            dst_dir = military_dir if pred == 1 else civilian_dir
            shutil.copy(path, os.path.join(dst_dir, os.path.basename(path)))

    return folder_path


def classify_labeled_folder(
    model,
    root_dir: str,
    threshold: float = DEFAULT_THRESHOLD
):
    transform = build_transform()
    dataset = datasets.ImageFolder(root=root_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_probs = []
    all_preds = []
    all_labels = []
    all_paths = [path for path, _ in dataset.samples]

    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(DEVICE)
            outputs = model(images).squeeze(1)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs >= threshold).astype(int)

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(targets.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    report = classification_report(
        y_true, y_pred,
        target_names=["civilian", "military"],
        digits=4,
        output_dict=False,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)

    # Опционально: сохранить ошибочные примеры
    errors_dir = os.path.join(root_dir, "errors")
    os.makedirs(errors_dir, exist_ok=True)
    fp_dir = os.path.join(errors_dir, "false_positive")  # civ -> mil
    fn_dir = os.path.join(errors_dir, "false_negative")  # mil -> civ
    os.makedirs(fp_dir, exist_ok=True)
    os.makedirs(fn_dir, exist_ok=True)

    for i, (true_label, pred_label) in enumerate(zip(all_labels, all_preds)):
        if true_label != pred_label:
            src_path = all_paths[i]
            filename = os.path.basename(src_path)
            if true_label == 0 and pred_label == 1:  # FP
                shutil.copy(src_path, os.path.join(fp_dir, filename))
            elif true_label == 1 and pred_label == 0:  # FN
                shutil.copy(src_path, os.path.join(fn_dir, filename))

    return report, cm