import os
import time
import random
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from PIL import Image, ImageDraw
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support

# =========================
# Настройки
# =========================

# Локальные пути (поменяй под себя)
DATASET_DIR = r"E:\projects\ground vehicle classification\vehicle_dataset"  # папка с civilian/ и military/
MODEL_OUT = r"E:\projects\ground vehicle classification\models\best_efficientnet_b0_aug2_newdataset.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EPOCHS = 5
LR = 1e-4
VAL_SPLIT = 0.2
NUM_WORKERS = 0  # для Windows лучше 0 или 1
INPUT_SIZE = 224

# =========================
# Аугментации
# =========================

class RandomGrayAug:
    def __init__(self, p_gray=0.3):
        self.p_gray = p_gray

    def __call__(self, img: Image.Image):
        if random.random() < self.p_gray:
            return img.convert('L').convert('RGB')
        return img


class FixedOcclusionRectsAug:
    def __init__(self, p=0.5, big_rects=2, big_w_frac=0.3, big_h_frac=0.4,
                 small_rects=1, small_w_frac=0.15, small_h_frac=0.15):
        self.p = p
        self.big_rects = big_rects
        self.big_w_frac = big_w_frac
        self.big_h_frac = big_h_frac
        self.small_rects = small_rects
        self.small_w_frac = small_w_frac
        self.small_h_frac = small_h_frac

    def _draw_rects(self, draw, w, h, n, w_frac, h_frac):
        rw = int(w_frac * w)
        rh = int(h_frac * h)
        for _ in range(n):
            x1 = random.randint(0, max(1, w - rw))
            y1 = random.randint(0, max(1, h - rh))
            x2 = x1 + rw
            y2 = y1 + rh
            color = (random.randint(20, 90), random.randint(60, 150), random.randint(10, 80))
            draw.rectangle([x1, y1, x2, y2], fill=color)

    def __call__(self, img: Image.Image):
        if random.random() > self.p:
            return img
        img = img.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size
        self._draw_rects(draw, w, h, self.big_rects, self.big_w_frac, self.big_h_frac)
        self._draw_rects(draw, w, h, self.small_rects, self.small_w_frac, self.small_h_frac)
        return img


train_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.RandomHorizontalFlip(),
    RandomGrayAug(p_gray=0.3),
    FixedOcclusionRectsAug(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def count_per_class_imagefolder(dataset):
    if isinstance(dataset, torch.utils.data.Subset):
        targets = [dataset.dataset.targets[i] for i in dataset.indices]
        classes = dataset.dataset.classes
    else:
        targets = dataset.targets
        classes = dataset.classes
    cnt = Counter(targets)
    return {classes[k]: v for k, v in sorted(cnt.items())}


def main_efficientnet_b0():
    # Если нужно, можно добавить распаковку архива локально, но обычно датасет уже лежит в DATASET_DIR
    # if not os.path.exists(DATASET_DIR):
    #     ...

    # Загрузка и разбиение данных
    full_dataset = datasets.ImageFolder(root=DATASET_DIR, transform=None)

    generator = torch.Generator().manual_seed(42)
    n_total = len(full_dataset)
    n_val = int(n_total * VAL_SPLIT)
    n_train = n_total - n_val

    train_subset, val_subset = random_split(
        full_dataset, [n_train, n_val], generator=generator
    )

    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform = val_transform

    # Веса классов
    train_labels = [full_dataset.targets[i] for i in train_subset.indices]
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(train_labels), y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=False
    )
    val_loader = DataLoader(
        val_subset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False
    )

    # Модель
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1].unsqueeze(0))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    # Обучение
    best_val_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).float().unsqueeze(1)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds.cpu() == labels.long().cpu()).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # Валидация
        model.eval()
        val_correct = 0
        val_total = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE).float().unsqueeze(1)
                logits = model(images)
                preds = (torch.sigmoid(logits) > 0.5).long()

                val_correct += (preds.cpu() == labels.long().cpu()).sum().item()
                val_total += labels.size(0)

                all_labels.extend(labels.cpu().numpy().ravel())
                all_preds.extend(preds.cpu().numpy().ravel())

        val_acc = val_correct / val_total
        scheduler.step(val_acc)

        prec, rec, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=None, labels=[0, 1], zero_division=0
        )

        print(
            f"Epoch {epoch}/{NUM_EPOCHS} | "
            f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | "
            f"F1(civ)={f1[0]:.4f}, F1(mil)={f1[1]:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "class_to_idx": full_dataset.class_to_idx,
                "val_acc": best_val_acc,
            }, MODEL_OUT)

    print(f"Лучшая val_acc: {best_val_acc:.4f}")
    if os.path.exists(MODEL_OUT):
        print(f"Размер модели: {os.path.getsize(MODEL_OUT) / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main_efficientnet_b0()